import argparse
from openai import OpenAI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        questions_input = f.read()

    if not questions_input:
        exit()

    client = OpenAI()

    response = client.responses.create(
      model="gpt-4o-mini",
      input=[
        {
          "role": "system",
          "content": [
            {
              "type": "input_text",
              "text": "The task is to process OCR output from a TV quiz show, correcting errors and removing duplications, and format each question and its answers as specified.\n\n# Steps\n\n1. **Extract Information**: Identify and extract each question and its associated answers from the OCR output.\n2. **Correction**: Correct any small errors in the text, such as typos or formatting issues.\n3. **Deduplication**: Remove any duplicate entries within the extracted questions and answers.\n4. **Classification**: Identify the correct answer for each question if it is not explicitly stated.\n5. **Formatting**: Format the corrected question and answers according to the specified layout.\n\n# Output Format\n\nThe output should be formatted for each question set as follows:\n\n```\nQuestion: [full question text]\nAnswer A: [answer text]\nAnswer B: [answer text]\nAnswer C: [answer text]\nCorrect answer: [Correct Answer Letter]\n===\n```\n\nEach question set should be separated by `===`.\n\n# Examples\n\n**Input:**\n$$$$\nQuestion\nm EEE ——\nWas sind die Teilnehmenden der ZDFneo-Show \"Glow Up\"?\n\n$$$\nAnswer A\nA:CPUs\n\n$$$\nAnswer B\nIUAS\n\n$$$\nAnswer C\nJFOS\n\n$$$\nCorrect answer: B\n\n\n**Output:**\n```\n===\nQuestion: Was sind die Teilnehmenden der ZDFneo-Show \"Glow Up\"?\nAnswer A: CPUs\nAnswer B: IUAs\nAnswer C JFOs\nCorrect answer: B\n```\n\n# Notes\n\n- Ensure that any recognized errors or duplications in the OCR text are corrected before formatting.\n- In cases where the correct answer isn't clearly indicated, use logical inference to determine the correct answer based on context."
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "input_text",
              "text": questions_input
            }
          ]
        }
      ],
      text={
        "format": {
          "type": "text"
        }
      },
      reasoning={},
      tools=[],
      temperature=1,
      max_output_tokens=2048,
      top_p=1,
      store=True
    )
    print(response.output_text)
    with open(args.output_file, "w") as f:
        f.write(
                response.output_text
                )

