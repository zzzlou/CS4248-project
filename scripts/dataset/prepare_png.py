import pandas as pd
import regex
import emoji
import json
import asyncio
from pyppeteer import launch


# # Read the CSV file
# df = pd.read_csv('/workspace/CS4248-project/data/elco/final_ELCo.csv')
# # Get unique emojis from EM column
# unique_emojis = set()

# for emoji_str in df['EM']:
#     emoji_list = emoji.emoji_list(emoji_str)
#     for emoji_item in emoji_list:
#         unique_emojis.add(emoji_item['emoji'])
# # for emoji_str in df['EM']:
# #     # 使用 \X 模式，遍历每个“扩展字形簇”
# #     clusters = regex.findall(r'\X', emoji_str)
# #     for cluster in clusters:
# #         # 如果该“字形簇”是 Emoji（或至少包含 Emoji），将它加入集合
# #         # 这里用 emoji 库的 UNICODE_EMOJI 判定最简单
# #         if any(ch in emoji.UNICODE_EMOJI['en'] for ch in cluster):
# #             unique_emojis.add(cluster)

# print("Unique emojis:", sorted(unique_emojis))

# # save to a json file
# with open('/workspace/CS4248-project/data/elco/unique_emojis.json', 'w') as f:
#     json.dump(list(unique_emojis), f)

async def emoji_to_png(emoji_str, filename):
    # 启动无头浏览器
    browser = await launch()
    page = await browser.newPage()
    
    # 在 HTML 中放大字体，以得到高分辨率
    # 可加一些内联CSS控制，比如 margin/padding/background 等
    content = f"""
    <html>
      <body style="font-size: 128px; margin: 0; padding: 0;">
        {emoji_str}
      </body>
    </html>
    """
    await page.setContent(content)
    
    # 让页面大小自适应
    await page.evaluate("""() => {
        document.body.style.width = 'auto';
        document.body.style.height = 'auto';
    }""")

    # 获取 body 元素，并截图
    body = await page.querySelector("body")
    await body.screenshot({'path': filename})

    await browser.close()

# 如果只想测试，可以这样直接跑：
emoji = "👨‍❤️‍💋‍👨"
asyncio.get_event_loop().run_until_complete(emoji_to_png(emoji, "emoji.png"))
