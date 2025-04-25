import argparse
import re
import sqlite3
import ssl
from urllib.request import urlopen

import cv2
import ffmpeg
import numpy as np
import pytesseract
from openai import OpenAI

ssl._create_default_https_context = ssl._create_stdlib_context

SRC_WIDTH = 960


def c_w(w):
    return int(w * SRC_WIDTH / 1920)


SRC_HEIGHT = 540


def c_h(h):
    return int(h * SRC_HEIGHT / 1080)


def probe_stream(url):
    res = ffmpeg.probe(url)

    def extract(entry):
        return {
            "index": entry["index"],
            "height": entry["height"],
            "width": entry["width"],
        }

    details = [extract(e) for e in res["streams"] if e["codec_type"] == "video"]
    return details


def select_stream(details, quality):
    target_height = {"low": [360, 540, 720, 1080], "high": [1080, 720, 540, 360]}
    for target in target_height[quality]:
        for detail in details:
            if detail["height"] == target:
                return detail
    raise Exception()


def process_stream(url, stream_index=0):
    proc = (
        ffmpeg.input(url)[f"{stream_index}"]
        .filter("fps", fps="2")
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(SRC_WIDTH, SRC_HEIGHT),
        )
        .run_async(pipe_stdout=True)
    )
    count = 0
    skip = 0
    result = []
    while True:
        in_bytes = proc.stdout.read(SRC_HEIGHT * SRC_WIDTH * 3)
        if not in_bytes:
            break
        if skip > 0:
            skip -= 1
            continue
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([SRC_HEIGHT, SRC_WIDTH, 3])
        # in_frame = in_frame[c_w(1405):c_h(240), c_w(258):c_h(798)]
        in_frame = in_frame[c_h(798) : c_h(1038), c_w(258) : c_w(1663)]
        img = in_frame[..., ::-1].copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_blur = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=0, sigmaY=0)
        sobel_h = cv2.Sobel(img_blur, dx=0, dy=1, ksize=5, ddepth=cv2.CV_64F)
        sobel_v = cv2.Sobel(img_blur, dx=1, dy=0, ksize=5, ddepth=cv2.CV_64F)

        horizontal_lines = (
            np.mean(sobel_h[c_h(0) : c_h(22), :])
            + np.mean(sobel_h[c_h(144) : c_h(160)])
            + np.mean(sobel_h[c_h(220) : c_h(226)])
        )
        vertical_lines = (
            np.mean(np.abs(sobel_v[:, : c_w(23)]))
            + np.mean(np.abs(sobel_v[c_h(156) : c_h(210), c_w(468) : c_w(480)]))
            + np.mean(np.abs(sobel_v[c_h(156) : c_h(210), c_w(920) : c_w(930)]))
            + np.mean(np.abs(sobel_v[:, c_w(1389) :]))
        )
        if horizontal_lines > 1000 and vertical_lines > 1000:  # question with answers
            ans = green_answer(img)
            if ans > -1:
                # _, tess_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                # tess_img = cv2.cvtColor(tess_img, cv2.COLOR_BGR2GRAY)
                mask_wrong = ((np.min(img, axis=-1) < 160) * 255).astype(np.uint8)
                mask_correct = ((np.max(img, axis=-1) > 60) * 255).astype(np.uint8)
                question = (
                    str(count)
                    + "  "
                    + pytesseract.image_to_string(
                        mask_wrong[c_h(20) : c_h(143), c_w(25) : c_w(1384)], lang="deu"
                    )
                )
                cv2.imwrite(f"slices/{count}.png", img)

                ans_sliceh = slice(c_h(156), c_h(217))
                ans1_slicew = slice(c_w(18), c_w(471))
                ans2_slicew = slice(c_w(486), c_w(920))
                ans3_slicew = slice(c_w(937), c_w(1384))

                ans_config = "--psm 7"

                if (
                    mask_correct[ans_sliceh, ans1_slicew].sum()
                    < mask_wrong[ans_sliceh, ans1_slicew].sum()
                ):
                    ans1 = pytesseract.image_to_string(
                        mask_correct[ans_sliceh, ans1_slicew],
                        lang="deu",
                        config=ans_config,
                    )
                else:
                    ans1 = pytesseract.image_to_string(
                        mask_wrong[ans_sliceh, ans1_slicew],
                        lang="deu",
                        config=ans_config,
                    )

                if (
                    mask_correct[ans_sliceh, ans2_slicew].sum()
                    < mask_wrong[ans_sliceh, ans2_slicew].sum()
                ):
                    ans2 = pytesseract.image_to_string(
                        mask_correct[ans_sliceh, ans2_slicew],
                        lang="deu",
                        config=ans_config,
                    )
                else:
                    ans2 = pytesseract.image_to_string(
                        mask_wrong[ans_sliceh, ans2_slicew],
                        lang="deu",
                        config=ans_config,
                    )

                if (
                    mask_correct[ans_sliceh, ans3_slicew].sum()
                    < mask_wrong[ans_sliceh, ans3_slicew].sum()
                ):
                    ans3 = pytesseract.image_to_string(
                        mask_correct[ans_sliceh, ans3_slicew],
                        lang="deu",
                        config=ans_config,
                    )
                else:
                    ans3 = pytesseract.image_to_string(
                        mask_wrong[ans_sliceh, ans3_slicew],
                        lang="deu",
                        config=ans_config,
                    )

                result.append((count, question, ans1, ans2, ans3, ans))
        else:
            skip = 1

        count += 1

    return result


def bgr2hsl(bgr):
    bgr /= 255.0
    dominant_color = np.argmax(bgr)
    det = np.max(bgr) - np.min(bgr)
    if det == 0:
        h = np.float32(0)
    elif dominant_color == 0:
        h = 4 + (bgr[2] - bgr[1]) / det
    elif dominant_color == 1:
        h = 2 + (bgr[0] - bgr[2]) / det
    else:
        h = (bgr[1] - bgr[0]) / det
    h *= 60

    l = (np.min(bgr) + np.max(bgr)) / 2
    if det == 0:
        s = np.float32(0)
    elif l > 0.5:
        s = det / (np.max(bgr) + np.min(bgr))
    else:
        s = det / (2.0 - np.max(bgr) - np.min(bgr))

    return h.item(), s.item(), l.item()


def green_answer(image):
    correct_a = bgr2hsl(
        np.mean(image[c_h(210) : c_h(215), c_w(45) : c_w(65)], axis=(0, 1))
    )
    correct_b = bgr2hsl(
        np.mean(image[c_h(210) : c_h(215), c_w(495) : c_w(515)], axis=(0, 1))
    )
    correct_c = bgr2hsl(
        np.mean(image[c_h(210) : c_h(215), c_w(945) : c_w(965)], axis=(0, 1))
    )

    def is_green(clr):
        return (
            clr[0] > 128
            and clr[0] < 138
            and clr[1] > 0.2
            and clr[1] < 0.32
            and clr[2] > 0.56
            and clr[2] < 0.69
        )

    if is_green(correct_a):
        return 0
    if is_green(correct_b):
        return 1
    if is_green(correct_c):
        return 2
    return -1


def to_string(results):
    correct_answers = ["A", "B", "C"]
    result = ""
    for r in results:
        result += "$$$$" + "\n"

        result += "Question\n"
        result += r[1] + "\n"

        result += "$$$" + "\n"
        result += "Answer A\n"
        result += r[2] + "\n"

        result += "$$$" + "\n"
        result += "Answer B\n"
        result += r[3] + "\n"

        result += "$$$" + "\n"
        result += "Answer C\n"
        result += r[4] + "\n"

        result += "$$$" + "\n"
        result += f"Correct answer: {correct_answers[r[5]]}\n\n"
    return result


def postprocess(s):
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": 'The task is to process OCR output from a TV quiz show, correcting errors and removing duplications, and format each question and its answers as specified.\n\n# Steps\n\n1. **Extract Information**: Identify and extract each question and its associated answers from the OCR output.\n2. **Correction**: Correct any small errors in the text, such as typos or formatting issues.\n3. **Deduplication**: Remove any duplicate entries within the extracted questions and answers.\n4. **Classification**: Identify the correct answer for each question if it is not explicitly stated.\n5. **Formatting**: Format the corrected question and answers according to the specified layout.\n\n# Output Format\n\nThe output should be formatted for each question set as follows:\n\n```\nQuestion: [full question text]\nAnswer A: [answer text]\nAnswer B: [answer text]\nAnswer C: [answer text]\nCorrect answer: [Correct Answer Letter]\n===\n```\n\nEach question set should be separated by `===`.\n\n# Examples\n\n**Input:**\n$$$$\nQuestion\nm EEE ——\nWas sind die Teilnehmenden der ZDFneo-Show "Glow Up"?\n\n$$$\nAnswer A\nA:CPUs\n\n$$$\nAnswer B\nIUAS\n\n$$$\nAnswer C\nJFOS\n\n$$$\nCorrect answer: B\n\n\n**Output:**\n```\n===\nQuestion: Was sind die Teilnehmenden der ZDFneo-Show "Glow Up"?\nAnswer A: CPUs\nAnswer B: IUAs\nAnswer C JFOs\nCorrect answer: B\n```\n\n# Notes\n\n- Ensure that any recognized errors or duplications in the OCR text are corrected before formatting.\n- In cases where the correct answer isn\'t clearly indicated, use logical inference to determine the correct answer based on context.',
                    }
                ],
            },
            {"role": "user", "content": [{"type": "input_text", "text": s}]},
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=True,
    )

    return response.output_text


def save_db(s, video_url):
    con, cur = open_db()
    re_question = re.compile(r"^Question: (.*)$", re.MULTILINE)
    re_answerA = re.compile(r"^Answer A: (.*)$", re.MULTILINE)
    re_answerB = re.compile(r"^Answer B: (.*)$", re.MULTILINE)
    re_answerC = re.compile(r"^Answer C: (.*)$", re.MULTILINE)
    re_correct = re.compile(r"^Correct answer: (.)$", re.MULTILINE)

    def extract_regex(s, regex):
        res = regex.search(s)
        if res is None:
            return None
        return res[1]

    for line in s.split("==="):
        question = ""
        ansA, ansB, ansC = "", "", ""
        correctAns = ""

        question = extract_regex(line, re_question)
        if question is None:
            print(f"Error with question extraction of '{line}'")
            continue

        ansA = extract_regex(line, re_answerA)
        if ansA is None:
            print(f"Error with answer A in {line}")
            continue

        ansB = extract_regex(line, re_answerB)
        if ansB is None:
            print(f"Error with answer B in {line}")
            continue

        ansC = extract_regex(line, re_answerC)
        if ansC is None:
            print(f"Error with answer C in {line}")
            continue

        correctAns = extract_regex(line, re_correct)
        if correctAns is None:
            print(f"Error with correct answer of {line}")
            continue

        cur.execute(
            """
            INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (None, question, ansA, ansB, ansC, correctAns, video_url),
        )

    con.commit()
    con.close()


def process_video(url):
    global SRC_WIDTH, SRC_HEIGHT
    streams = probe_stream(url)
    high_q_stream = select_stream(streams, "high")
    SRC_HEIGHT = high_q_stream["height"]
    SRC_WIDTH = high_q_stream["width"]
    res = process_stream(url, high_q_stream["index"])
    str_res = to_string(res)
    postprocessed = postprocess(str_res)
    save_db(postprocessed, url)


def read_html(url):
    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    return html


def get_video_from_html(url):
    html = read_html(url)
    pos = html.index("uploadDate")
    upload_time = html[pos + 21 : pos + 41]  # not used yet
    pos = html.index("m3u8")
    start_pos = html.rfind('"', 0, pos)
    video_url = html[start_pos + 1 : pos + 4]
    return video_url


def get_indiv_urls_from_overview(html, needle, prefix):
    urls = []
    pos = html.find(needle)
    while pos > -1:
        start = html.rfind('"', 0, pos)
        end = html.find('"', pos)
        pos = html.find(needle, pos + 1)
        video_url = get_video_from_html(prefix + html[start + 1 : end])
        urls.append(video_url)
    return urls


def open_db():
    con = sqlite3.connect("questions.db")
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS questions (id INTEGER PRIMARY KEY AUTOINCREMENT,  question UNIQUE, answer_a, answer_b, answer_c, correct_answer, url)"
    )
    con.commit()
    return con, cur


def in_db(url):
    con, cur = open_db()
    cur.execute("SELECT id FROM questions WHERE url=?", (url,))
    res = cur.fetchone()
    con.close()

    if res:
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    if args.video.startswith("http"):
        video_url = get_video_from_html(args.video)
        if not in_db(video_url):
            process_video(video_url)
    elif args.video.endswith(".yaml"):
        import yaml

        with open(args.video, "r") as file:
            config = yaml.safe_load(file)

        html = read_html(config["url"])
        urls = get_indiv_urls_from_overview(
            html, config["searchTerm"], config["prefix"]
        )
        for url in urls:
            if not in_db(url):
                process_video(url)
    else:
        if not in_db(args.video):
            process_video(args.video)
