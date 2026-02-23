# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Coroutine
from logging.handlers import RotatingFileHandler

import aiohttp
import emoji
from google import genai
from google.genai import types as genai_types
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem
from azure.core.exceptions import HttpResponseError
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from lingua import LanguageDetectorBuilder, Language
from telethon import events
from telethon.sync import TelegramClient
from telethon.tl.types import MessageEntityBlockquote

workspace = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

# Create logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

# Create handler for log file
handler = RotatingFileHandler(f'{workspace}/log.txt', maxBytes=20000000, backupCount=5)

# Define output format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)

# Create handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(stream_handler)

detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
all_langs = Language.all()
all_langs = {lang.iso_code_639_1.name.lower(): lang.name for lang in all_langs}


def load_config() -> dict:
    # load config from json file, check if the file exists first
    if not os.path.exists(f'{workspace}/config.json'):
        logger.error('config.json not found, created an empty one')
        sys.exit(1)

    with open(f'{workspace}/config.json', 'r') as f:
        config = json.load(f)

    return config


def save_config() -> None:
    cfg['target_config'] = target_config
    with open(f'{workspace}/config.json', 'w') as f:
        json.dump(cfg, f, indent=2)


## configuration
cfg = load_config()
## telegram config
api_id = cfg['api_id']
api_hash = cfg['api_hash']
## Block quote will be collapsed if the length of the text exceeds this value
collapsed_length = cfg.get('collapsed_length', 0)
## translation service
translation_service = cfg['translation_service']
## google config
google_config = cfg.get('google', {})
google_creds = google_config.get('creds', '')
## azure config
azure_config = cfg.get('azure', {})
azure_key = azure_config.get('key', '')
azure_endpoint = azure_config.get('endpoint', '')
azure_region = azure_config.get('region', '')
## deeplx config
deeplx_config = cfg.get('deeplx', {})
deeplx_url = deeplx_config.get('url', 'https://api.deeplx.org/translate')
## openai config
openai_config = cfg.get('openai', {})
openai_api_key = openai_config.get('api_key', '')
openai_url = openai_config.get('url', 'https://api.openai.com/v1/chat/completions')
openai_model = openai_config.get('model', 'gpt-3.5-turbo')
openai_prompt = openai_config.get('prompt', '')
openai_temperature = openai_config.get('temperature', 0.5)
## gemini config
gemini_config = cfg.get('gemini', {})
gemini_api_key = gemini_config.get('api_key', '')
gemini_model = gemini_config.get('model', '')
gemini_prompt = gemini_config.get('prompt', '')
gemini_temperature = gemini_config.get('temperature', 0.5)
## target config
target_config = cfg.get('target_config', {})

# Initialize Telegram client
client = TelegramClient(f'{workspace}/client', api_id, api_hash)

# Google Translation Service Initialization
if translation_service == 'google':
    if not google_creds:
        logger.error("Google translation service configuration is missing")
        sys.exit(1)
    google_credentials = service_account.Credentials.from_service_account_info(google_creds)
    google_client = translate.Client(credentials=google_credentials)

# Azure Translation Service Initialization
if translation_service == 'azure':
    if not azure_key or not azure_endpoint or not azure_region:
        logger.error("Azure translation service configuration is missing")
        sys.exit(1)
    text_translator = TextTranslationClient(endpoint=azure_endpoint,
                                            credential=TranslatorCredential(azure_key, azure_region))

if translation_service == 'gemini':
    if not gemini_config or not gemini_api_key:
        logger.error("Gemini translation service configuration is missing")
    gemini_client = genai.Client(api_key=gemini_api_key)


def remove_links(text: str) -> str:
    # regrex pattern for URL
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # use re.sub to remove the URL from the text
    return re.sub(url_pattern, '', text).strip()


_TRANSLATION_DISPATCH: dict[str, Callable[..., Coroutine]] = {}


async def translate_text(text: str, source_lang: str, target_langs: list[str]) -> dict[str, str]:
    result = {}
    if emoji.purely_emoji(text):
        return result
    detect_lang = detector.detect_language_of(text).iso_code_639_1.name.lower()
    if detect_lang in target_langs and detect_lang != source_lang:
        return result

    translate_fn = _TRANSLATION_DISPATCH.get(translation_service)
    if translate_fn is None:
        raise ValueError(
            f"Unknown translation service: {translation_service}. "
            f"Available: {', '.join(_TRANSLATION_DISPATCH)}")

    async with aiohttp.ClientSession() as session:
        text_without_link = remove_links(text)
        # Execute translation tasks concurrently
        async with asyncio.TaskGroup() as tg:
            task_handles = []
            for target_lang in target_langs:
                if source_lang == target_lang:
                    result[target_lang] = text
                    continue
                task_handles.append(tg.create_task(
                    translate_fn(text_without_link, source_lang, target_lang, session)
                ))
        for handle in task_handles:
            lang, translated = handle.result()
            result[lang] = translated
    return result


# Google Translation API
async def translate_google(text: str, source_lang: str, target_lang: str, session: aiohttp.ClientSession) -> tuple[str, str]:
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    result = google_client.translate(text, target_language=target_lang, format_='text')
    logger.info(f'Text: {result["input"]}')
    logger.info(f'Translation: {result["translatedText"]}')
    logger.info(f'Detected source language: {result["detectedSourceLanguage"]}')

    return target_lang, result["translatedText"]


# DeepLX Translation API
async def translate_deeplx(text: str, source_lang: str, target_lang: str, session: aiohttp.ClientSession) -> tuple[str, str]:
    url = deeplx_url
    payload = {
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    start_time = time.time()
    async with session.post(url, json=payload) as response:
        logger.info(f"Translation from {source_lang} to {target_lang} took: {time.time() - start_time}")
        if response.status != 200:
            logger.error(f"Translation failed: {response.status}")
            raise RuntimeError(f"Translation failed: {response.status}")

        result = await response.json()
        if result['code'] != 200:
            logger.error(f"Translation failed: {result}")
            raise RuntimeError(f"Translation failed: {result}")

    return target_lang, result['data']


# Azure Translation API
async def translate_azure(text: str, source_lang: str, target_lang: str, session: aiohttp.ClientSession) -> tuple[str, str]:
    try:
        source_language = source_lang
        target_languages = [target_lang]
        input_text_elements = [InputTextItem(text=text)]

        response = text_translator.translate(body=input_text_elements, to_language=target_languages,
                                             from_language=source_language)
        translation = response[0] if response else None

        if translation:
            for translated_text in translation.translations:
                logger.info(
                    f"Text was translated to: '{translated_text.to}' and the result is: '{translated_text.text}'.")
                return translated_text.to, translated_text.text

    except HttpResponseError as exception:
        if exception.error is not None:
            logger.error(f"Error Code: {exception.error.code}")
            logger.error(f"Message: {exception.error.message}")
        raise


# OpenAI Translation API
async def translate_openai(text: str, source_lang: str, target_lang: str, session: aiohttp.ClientSession) -> tuple[str, str]:
    url = openai_url
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    prompt = openai_prompt.replace('tgt_lang', all_langs.get(target_lang, target_lang))
    text = "Source Text: \n" + text
    logger.debug(f"Prompt: {prompt}")
    payload = {
        'messages': [
            {
                'role': 'system',
                'content': prompt,
            },
            {
                'role': 'user',
                'content': text,
            }
        ],
        'stream': False,
        'model': openai_model,
        'temperature': openai_temperature,
        'presence_penalty': 0,
        'frequency_penalty': 0,
        'top_p': 1,
        'thinking': {'type': 'enabled', 'budget_tokens': 0}
    }

    start_time = time.time()
    async with session.post(url, headers=headers, data=json.dumps(payload)) as response:
        logger.info(f"Translation from {source_lang} to {target_lang} took: {time.time() - start_time}")
        response_text = await response.text()
        result = json.loads(response_text)
        try:
            return target_lang, result['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"OpenAI translation failed: {response_text} {e}")


async def translate_gemini(text: str, source_lang: str, target_lang: str, session: aiohttp.ClientSession) -> tuple[str, str]:
    prompt = gemini_prompt.replace('tgt_lang', all_langs.get(target_lang, target_lang))
    response = gemini_client.models.generate_content(
        model=gemini_model,
        contents=text,
        config=genai_types.GenerateContentConfig(
            system_instruction=prompt,
            temperature=gemini_temperature,
            safety_settings=[genai_types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF",
            )],
        ),
    )
    return target_lang, response.text.strip()


_TRANSLATION_DISPATCH.update({
    'openai': translate_openai,
    'gemini': translate_gemini,
    'google': translate_google,
    'azure': translate_azure,
    'deeplx': translate_deeplx,
})


async def command_mode(event: events.NewMessage.Event, target_key: str, text: str) -> None:
    if text.startswith('.tt-on-global') or text == '.tt-off-global':
        target_key = f'0.{event.sender_id}'
        text = text.replace('-global', '')

    if text == '.tt-off':
        await event.delete()
        if target_key in target_config:
            del target_config[target_key]
            save_config()
            logger.info(f"Disabled: {target_key}")
        return

    if text.startswith('.tt-on,'):
        _, source_lang, target_langs = text.split(',')
        if not source_lang or not target_langs:
            await event.message.edit("Invalid command, correct format: .tt-on,source_lang,target_lang1|target_lang2")
        else:
            target_config[target_key] = {
                'source_lang': source_lang,
                'target_langs': target_langs.split('|')
            }
            save_config()
            logger.info(f"Settings applied: {target_config[target_key]}")
            await event.message.edit(f"Settings applied: {target_config[target_key]}")
        await asyncio.sleep(3)
        await event.message.delete()
        return

    if text.startswith('.tt-skip'):
        await event.message.edit(text[8:].strip())
        logger.info("Skipped translation")
        return

    if text.startswith('.tt-once,'):
        command, raw_text = text.split(' ', 1)
        _, source_lang, target_langs = command.split(',')
        logger.info(f"Translating message: {raw_text}")
        await translate_and_edit(event.message, raw_text, source_lang, target_langs.split('|'))
        return

    await event.message.edit("Unknown command")
    await asyncio.sleep(3)
    await event.message.delete()


# Listen for new and edited outgoing messages
@client.on(events.NewMessage(outgoing=True))
@client.on(events.MessageEdited(outgoing=True))
async def handle_message(event: events.NewMessage.Event) -> None:
    target_key = f'{event.chat_id}.{event.sender_id}'
    try:
        message = event.message
        # Ignore empty messages
        if not message.text:
            return
        message_content = message.text.strip()
        if not message_content:
            return

        # skip PagerMaid commands
        if message_content.startswith(','):
            return

        # skip bot commands
        if message_content.startswith('/'):
            return

        # command mode
        if message_content.startswith('.tt-'):
            await command_mode(event, target_key, message_content)
            return

        # handle reply message
        if message.reply_to_msg_id and message_content.startswith('.tt,'):
            _, source_lang, target_langs = message_content.split(',')
            logger.info(f"Reply message: {message.reply_to_msg_id}")
            reply_message = await client.get_messages(event.chat_id, ids=message.reply_to_msg_id)
            if not reply_message.text:
                return
            message_content = reply_message.text.strip()
            if source_lang and target_langs:
                logger.info(f"Translating message: {message.text}")
                await translate_and_edit(message, message_content, source_lang, target_langs.split('|'))
            return

        # handle edited message
        if isinstance(event, events.MessageEdited.Event):
            if message_content.startswith('.tt'):
                message_content = message_content[3:].strip()
            else:
                return

        # chat config
        config = {}
        if target_key in target_config:
            config = target_config[target_key]
        else:
            # global config
            target_key = f'0.{event.sender_id}'
            if target_key not in target_config:
                return
            config = target_config[target_key]

        logger.info(f"Translating message: {message.text}")
        source_lang = config['source_lang']
        target_langs = config['target_langs']
        await translate_and_edit(message, message_content, source_lang, target_langs)

    except Exception as e:
        # Log exception during message handling
        logger.exception("Error handling message")


async def translate_and_edit(message, message_content: str, source_lang: str, target_langs: list[str]) -> None:
    start_time = time.time()  # Record start time
    translated_texts = await translate_text(message_content, source_lang, target_langs)
    logger.info(f"Translation took: {time.time() - start_time}")

    if not translated_texts:
        return

    # Build translation text from all target languages
    all_translations = []
    for lang in target_langs:
        if lang in translated_texts:
            all_translations.append(translated_texts[lang])
    translation_text = '\n'.join(all_translations)

    # Keep original text and append translation as blockquote
    modified_message = message_content + '\n' + translation_text

    # Handle special characters such as emojis and other unicode characters
    pattern = u'[\U00010000-\U0010ffff]'
    pattern_matches_original = len(re.findall(pattern, message_content))
    pattern_matches_translation = len(re.findall(pattern, translation_text))

    # Blockquote starts after original text + newline
    offset = len(message_content) + pattern_matches_original + 1
    length = len(translation_text) + pattern_matches_translation

    if collapsed_length > 0 and length > collapsed_length:
        formatting_entities = [MessageEntityBlockquote(offset=offset, length=length, collapsed=True)]
    else:
        formatting_entities = [MessageEntityBlockquote(offset=offset, length=length)]

    # Edit the message
    await client.edit_message(message, modified_message, formatting_entities=formatting_entities)


# Start client and run until disconnected
try:
    client.start()
    client.run_until_disconnected()
finally:
    # Disconnect client
    client.disconnect()
