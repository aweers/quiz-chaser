import sqlite3
from contextlib import asynccontextmanager
from typing import Union

import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI

from chaser import get_indiv_urls_from_overview, in_db, process_video, read_html


async def cronjob():
    print("Cronjob")
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    html = read_html(config["url"])
    urls = get_indiv_urls_from_overview(html, config["searchTerm"], config["prefix"])
    for url in urls:
        if not in_db(url):
            process_video(url)


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
    with sqlite3.connect("questions.db") as conn:
        cur = conn.cursor()
        cur.execute("select count(1) from questions")
        rows = cur.fetchone()
        return rows[0]



@app.get("/questions")
def read_item(after: Union[int, None] = None):
    result = []
    with sqlite3.connect("questions.db") as conn:
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
                    "src": row[5],
                }
            )
    return result
