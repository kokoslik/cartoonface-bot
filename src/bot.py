import asyncio

from aiogram import Bot, Dispatcher, types, executor
from PIL import Image
from io import BytesIO
from model.generator import Generator
from torchvision import transforms as tt
import torch


# Insert your bot token and the correct pth file below
TOKEN = ''
PTH_FILE = 'dumps/genB_epoch199.pth'

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
q = asyncio.Queue()


async def worker(q):
    means = [0.5, 0.5, 0.5]
    stds = [0.5, 0.5, 0.5]
    pic_size=128
    transform = tt.Compose([tt.Resize(pic_size), tt.CenterCrop(pic_size), tt.ToTensor(), tt.Normalize(means, stds)])
    inv = tt.ToPILImage()
    gen = Generator(instance_norm=True)
    gen.load_state_dict(torch.load(PTH_FILE))
    gen.eval()
    while True:
        message = await q.get()
        byt = BytesIO()
        await message.photo[-1].download(destination_file=byt)
        image = Image.open(byt).convert('RGB')
        tensor = transform(image)
        with torch.no_grad():
            output = gen(tensor[None,:,:,:])
        reply = inv(output.squeeze()*0.5+0.5)
        byt = BytesIO()
        reply.save(byt, 'JPEG')
        byt.seek(0)
        await bot.send_photo(message.from_user.id, byt)


@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Welcome to CartooFaceBot! Send me a picture of your face, and I'll try to turn you into a cartoon :-)")


@dp.message_handler(content_types=['photo'])
async def process_photo(message: types.Message):
    await q.put(message)
    await message.reply("Photo added to queue")


@dp.message_handler()
async def process_any_message(message: types.Message):
    await message.reply("Send a pic and I'll do my best")


async def on_startup(smt):
    asyncio.create_task(worker(q))

if __name__ == '__main__':
    executor.start_polling(dp, on_startup=on_startup)
