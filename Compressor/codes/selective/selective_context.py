from selenium import webdriver
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
driver = webdriver.Chrome()
driver.get("https://huggingface.co/spaces/liyucheng/selective_context")
# 等待并切换到 iframe
WebDriverWait(driver, 10).until(
    EC.frame_to_be_available_and_switch_to_it((By.ID, "iFrameResizer0"))
)
def read_jsonl_file(file_path):
    """
    Load lines of texts.

    Args:
        file_path (str): Path for lines of texts.

    Returns:
        (List[str]): List of texts.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line.strip())
            data.append(json_data)
    return data

def write_jsonl_file(file_path, data):
    """
    Write list of JSON objects to a JSONL file.

    Args:
        file_path (str): Path for output file.
        data (List[dict]): List of JSON objects.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

def compress_text(input_text):
    """
    输入文本并获取压缩结果。

    Args:
        input_text (str): 输入的文本。

    Returns:
        str: 压缩后的文本。
    """
    try:
        input_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="root"]/div[1]/div[1]/div/div/div/section/div[1]/div[1]/div/div[10]/div/div[1]/div/div/textarea'))
        )
        # 清空文本框
        input_element.click()
        time.sleep(0.1)
        action_chains = ActionChains(driver)
        action_chains.key_down(Keys.COMMAND).send_keys('a').key_up(Keys.COMMAND).send_keys(Keys.DELETE).perform()
        time.sleep(0.2)
        # 输入新内容
        input_element.send_keys(input_text)
        print("raw:" + input_text)
    except Exception as e:
        print("输入失败:", e)
        driver.save_screenshot("error.png")
        return None
    time.sleep(2)

    try:
        compress_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="root"]/div[1]/div[1]/div/div/div/section/div[1]/div[1]/div/div[11]/div/button/div/p'))
        )
        compress_button.click()
        print("成功完成压缩")
    except Exception as e:
        print("压缩失败:", e)
        driver.save_screenshot("error.png")
        return None

    time.sleep(10)  # 等待压缩完成

    try:
        compress_text = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="root"]/div[1]/div[1]/div/div/div/section/div[1]/div[1]/div/div[14]/div/div/div/pre/div'))
        ).text
        print("成功获取压缩内容")
        print("compress:"+compress_text)
        return compress_text
    except Exception as e:
        print("获取压缩内容失败:", e)
        driver.save_screenshot("error.png")
        return None


def main(input_file_path, output_file_path):
    # 读取 JSONL 文件
    data = read_jsonl_file(input_file_path)
    compressed_data = []
    old_context = None
    for item in data:
        context = item.get('context', '')
        id = item.get('id', '')
        if id == 1:
            if context:
                compressed_context = compress_text(context)
                if compressed_context:
                    item['context'] = compressed_context
                    old_context = compressed_context
                    compressed_data.append(item)
                else:
                    print("error, context没获取到")
        else:
            item['context'] = old_context
            compressed_data.append(item)

    # 写入新的 JSONL 文件
    write_jsonl_file(output_file_path, compressed_data)
    print("压缩完成，结果已保存到文件中。")
# 现在可以操作 iframe 内的元素（如点击 "zh"）
# try:
#     zh_button = WebDriverWait(driver, 10).until(
#         EC.element_to_be_clickable((By.XPATH, "//div[text()='zh']"))
#     )
#     zh_button.click()
#     print("成功切换到中文")
# except Exception as e:
#     print("切换失败:", e)
#     driver.save_screenshot("error.png")

# try:
#     zh_button = WebDriverWait(driver, 10).until(
#         EC.element_to_be_clickable((By.XPATH, "//div[text()='0.5']"))
#     )
#     zh_button.click()
#     print("成功切换到中文")
# except Exception as e:
#     print("切换失败:", e)
#     driver.save_screenshot("error.png")


# for text in ['UZI IS THE BEST ADC IN THE WORLD', 'UZI IS THE WORST ADC IN THE WORLD']:
#         try:
#             input_element = WebDriverWait(driver, 10).until(
#                 EC.presence_of_element_located((By.XPATH,
#                                                 '//*[@id="root"]/div[1]/div[1]/div/div/div/section/div[1]/div[1]/div/div[10]/div/div[1]/div/div/textarea'))
#             )
#             # 清空文本框
#             input_element.click()
#             time.sleep(0.1)
#
#             # 全选并删除输入框内容
#             action_chains = ActionChains(driver)
#             action_chains.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).send_keys(Keys.DELETE).perform()
#             time.sleep(0.2)
#
#             # 输入新内容
#             input_element.send_keys(text)
#         except Exception as e:
#             print("输入失败:", e)
#             driver.save_screenshot("error.png")
#
#         try:
#             compress_button = WebDriverWait(driver, 10).until(
#                 EC.element_to_be_clickable((By.XPATH, '//*[@id="root"]/div[1]/div[1]/div/div/div/section/div[1]/div[1]/div/div[11]/div/button/div/p'))
#             )
#             compress_button.click()
#             print("成功完成压缩")
#         except Exception as e:
#             print("压缩失败:", e)
#             driver.save_screenshot("error.png")
#
#         time.sleep(1)
#
#         try:
#             compress_text = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH ,'//*[@id="root"]/div[1]/div[1]/div/div/div/section/div[1]/div[1]/div/div[14]/div/div/div/pre/div/code/span'))).text
#             print("成功复制内容")
#             print(compress_text)
#         except Exception as e:
#             print("输入失败:", e)
#             driver.save_screenshot("error.png")

if __name__ == "__main__":
    input_file_path = '../finetuning/datasets/test.jsonl'  # 输入文件路径
    output_file_path = '../finetuning/datasets/selective_0.5_train_new.jsonl'  # 输出文件路径
    main(input_file_path, output_file_path)

    # 等待用户手动关闭浏览器
    input("按任意键退出程序...")