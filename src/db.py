import sqlite3
import re

input_file = "llm.txt"

with open(input_file, "r") as f:
    llm = f.read()

con = sqlite3.connect("questions.db")
cur = con.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS questions (id INTEGER PRIMARY KEY AUTOINCREMENT,  question UNIQUE, answer_a, answer_b, answer_c, correct_answer, url)")

re_question = re.compile(r'^Question: (.*)$', re.MULTILINE)
re_answerA = re.compile(r'^Answer A: (.*)$', re.MULTILINE)
re_answerB = re.compile(r'^Answer B: (.*)$', re.MULTILINE)
re_answerC = re.compile(r'^Answer C: (.*)$', re.MULTILINE)
re_correct = re.compile(r'^Correct answer: (.)$', re.MULTILINE)

url = llm.split('\n')[0]

def extract_regex(s, regex):
    res = regex.search(s)
    if res is None:
        return None
    return res[1]

for line in llm.split("==="):
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

    cur.execute("""
        INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (None, question, ansA, ansB, ansC, correctAns, url))

con.commit()
con.close()
