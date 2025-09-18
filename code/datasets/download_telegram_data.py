import pandas as pd
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
import tempfile
import nest_asyncio
import asyncio
POST_COUNT = 3000

# Telegram
nest_asyncio.apply()
api_id = '20973948'
api_hash = 'de82e7ed090a4c44e8175f17e29dd6bc'
phone = '89874195267'
username = 'diana_minn'
session_file = tempfile.NamedTemporaryFile().name
client = TelegramClient(session_file, api_id, api_hash)

async def main():
    await client.start(phone)
    channel = await client.get_entity('goldapple_ru')
    offset_id = 0
    all_messages = []
    total_messages = 0
    while True:
        history = await client(GetHistoryRequest(
            peer = channel,
            offset_id = offset_id,
            offset_date = None,
            add_offset = 0,
            limit = 100,
            max_id = 0,
            min_id = 0,
            hash = 0
        ))
        if not history.messages:
            break
        messages = history.messages
        for message in messages:
            post = {
                'id': message.id,
                'date': message.date,
                'text': message.message,
                'views': message.views,
                'forwards': message.forwards,
                'replies': message.replies.replies if message.replies else 0,
                'reactions': {str(reaction.reaction): reaction.count for reaction in
                              message.reactions.results} if message.reactions else {}
            }
            all_messages.append(post)
            offset_id = message.id
        total_messages = len(all_messages)
        if POST_COUNT and total_messages >= POST_COUNT:
            break
    pd.DataFrame(all_messages).to_csv('data/raw/telegram_data.csv', index = False)

async def run():
    async with client:
        await main()

loop = asyncio.get_event_loop()
loop.run_until_complete(run())