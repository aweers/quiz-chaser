import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Union

import telegram
import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from chaser import get_indiv_urls_from_overview, in_db, process_video, read_html
from fastapi import FastAPI


async def send_message(msg, config):
    bot = telegram.Bot(token=config["TELEGRAM_TOKEN"])
    async with bot:
        await bot.send_message(
            chat_id=config["TELEGRAM_CHANNEL_ID"], text=f"[QUIZZZ] {msg}"
        )


async def cronjob():
    print("Cronjob")
    with open(os.path.join(Path(__file__).parent, "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    html = read_html(config["url"])
    urls = get_indiv_urls_from_overview(html, config["searchTerm"], config["prefix"])
    for url in urls:
        if not in_db(url):
            process_video(url)

    last_q = latest_question()
    await send_message(
        f"{last_q[1]}\n\nA: {last_q[2]}\nB: {last_q[3]}\nC: {last_q[4]}\n\nCorrect: {last_q[5]}\nTotal: {count()}",
        config,
    )


def latest_question():
    with sqlite3.connect("questions.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM questions ORDER BY id DESC LIMIT 1")
        result = cur.fetchone()
        return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = AsyncIOScheduler()
    trigger = CronTrigger(
        year="*", month="*", day="*", hour="15", minute="12", second="0"
    )
    scheduler.add_job(func=cronjob, trigger=trigger)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/count")
def count():
    with sqlite3.connect(os.path.join(Path(__file__).parent, "questions.db")) as conn:
        cur = conn.cursor()
        cur.execute("select count(1) from questions")
        rows = cur.fetchone()
        return rows[0]


@app.get("/trigger")
async def trigger():
    await cronjob()
    return 200


@app.get("/questions")
def read_item(after: Union[int, None] = None):
    result = []
    with sqlite3.connect(os.path.join(Path(__file__), "questions.db")) as conn:
        cur = conn.cursor()
        if after is None:
            after = 0
        cur.execute("select * from questions where id >= ?", (after,))
        rows = cur.fetchall()
        for row in rows:
            result.append(
                {
                    "id": row[0],
                    "q": row[1],
                    "a": row[2],
                    "b": row[3],
                    "c": row[4],
                    "correct": row[5],
                    "src": row[6],
                }
            )
    return result
