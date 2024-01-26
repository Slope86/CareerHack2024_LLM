import os
import re
import time

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from lc_module import *

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


# def save_embeddings(embeddings, filename):
#     with open(filename, "wb") as file:
#         pickle.dump(embeddings, file)


# def load_embeddings(filename):
#     with open(filename, "rb") as file:
#         return pickle.load(file)


# def embedding_all_doc():
#     filename = "embeddings.pkl"

#     # 檢查pickle檔案是否存在
#     if os.path.exists(filename):
#         # 從pickle檔案加載數據
#         total_db = load_embeddings(filename)
#         print("Loaded embeddings from file.")
#     else:
#         # 沒有pickle檔案，執行正常流程
#         total_db = _embedding_all_doc()

#         # 保存數據到pickle檔案
#         save_embeddings(total_db, filename)
#         print(f"Embeddings saved to {filename}")

#     return total_db


def embedding_all_doc():
    # 在雲端時記得改 ../
    url_cpu = "CPU utilization.pdf"
    db_cpu = doc2vector(url_cpu)
    time.sleep(10)
    url_Memory = "Memory utilization.pdf"
    db_Memory = doc2vector(url_Memory)
    time.sleep(10)
    url_Cloud = "Cloud run restart.pdf"
    db_Cloud = doc2vector(url_Cloud)
    time.sleep(10)
    url_Instance = "Instance count.pdf"
    db_Instance = doc2vector(url_Instance)
    time.sleep(10)
    url = "http_1_3.pdf"
    db = doc2vector(url)
    time.sleep(10)
    url2 = "http_4.pdf"
    db2 = doc2vector(url2)
    time.sleep(10)
    url3 = "http_5.pdf"
    db3 = doc2vector(url3)
    print("embedding finish")
    total_db = [db_cpu, db_Memory, db_Cloud, db_Instance, db, db, db, db2, db3]
    return total_db


total_db = embedding_all_doc()


def get_function_openai(inputdata: str = "Hello") -> list:
    """
    用來接收前端使用者的輸入，先判斷是屬於哪一種類型
     Parameters:
    - inputdata (str): 使用者輸入
    return:
    - list: ['class','arg1','arg2','arg3']

    """
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """
                Our GCP system has the following functions:
                1. GPT Q&A
                2. Analysis of historical logs for the past n days
                3. CPU upscaling
                4. Memory upscaling
                5. Real-time data monitoring
                6. send email
                You will receive various inputs from users, and your task is to determine which category the task belongs to and output it in the format ['function code', 'arg 1', 'arg 2', 'arg 3']:
                Here are some examples:
                User input: 'How are you?'
                Output: [1, How are you?, -1, -1] 
                Different functions require different parameters. You must determine what to fill in for each parameter. Additionally, avoid generating any extra text for ease of programming:
                Example 1:
                User input: 'I want to understand HTTP CODE'
                Output: [1,I want to understand HTTP CODE, -1, -1] ,Argument Description for instruction code 1: arg 1:user_query,arg 2:not use,alway -1 ,arg 3 : not use,always -1
                Example 2:
                User input: 'Check if there have been any anomalies in my system in the past 5 days, 30 hours, and 70 minutes'
                Output: [2, 5, 30, 70]
                Note: No need to convert time units. ,Argument Description for instruction code 2: arg 1: days (default -1 = None),arg 2:hours (default -1 = None) ,arg 3 : minutes (default -1 = None)
                Example 3:
                User input: 'Check if there have been any anomalies in my system in the past 30 hours'
                Output: [2, -1, 30, -1]
                Note:No need to convert time units. ,Argument Description for instruction code 2: arg 1: days (default -1 = None),arg 2:hours (default -1 = None) ,arg 3 : minutes (default -1 = None)
                Example 4:
                User input: 'Please find the latest information on memory from the past few days.'
                Output: [2, -1, 5, -1]
                Note: If no specific time is designated, the default is 5 hour.Argument Description for instruction code 2: arg 1: days (default -1 = None),arg 2:hours (default -1 = None) ,arg 3 : minutes (default -1 = None)
                Example 5:
                user_input: 'My CPU space is not enough, please increase the CPU capacity '
                output: [3, add, -1, -1]
                Note: Since it's a request to increase the CPU, arg1 = add,Argument Description for instruction code 3: arg 1: action of cpu (arg1 must in set(add,sub) ),arg 2:not use always -1 ,arg 3 : not use always -1
                Example 6:
                user_input: 'I have too much CPU space, reduce 2 CPUs for me'
                output: [3, sub, -1, -1] , arg1 = add,Argument Description: arg 1: action of cpu (arg1 must in set(add,sub) ),arg 2:not use always -1 ,arg 3 : not use always -1
                Note: Since it's a request to reduce the CPU, arg1 = sub. Additionally,Argument Description for instruction code 3: arg 1: action of cpu (arg1 must in set(add,sub) ),arg 2:not use always -1 ,arg 3 : not use always -1
                Example 7:
                user_input: 'My memory space is not enough, please increase the memory capacity by 128m'
                output: [4, add, -1, -1]
                Note:,Argument Description for instruction code 4: arg 1: action of memory (arg1 must in set(add,sub) ),arg 2:not use always -1 ,arg 3 : not use always -1
                Example 8:
                user_input: 'My memory space is overflowing, please reduce the memory capacity'
                output: [4, sub, -1, -1]
                Note:,Argument Description for instruction code 4: arg 1: action of memory (arg1 must in set(add,sub) ),arg 2:not use always -1 ,arg 3 : not use always -1
                Example 9:
                user_input: 'I need you to help me manage my system'
                output: [5, start, -1, -1]
                Note: since start manage system,arg1=start.Argument Description for instruction code 5: arg 1: action of manage  (arg1 must in set(start,stop) ),arg 2:not use always -1 ,arg 3 : not use always -1
                Example 10:
                user_input: 'You can stop helping me manage the system now'
                output: [5, stop, -1, -1]"
                Note: since start manage system,arg1=stop,.Argument Description for instruction code 5: arg 1: action of manage  (arg1 must in set(start,stop) ),arg 2:not use always -1 ,arg 3 : not use always -1
                Example 11:
                user_input: Please send an email to notify the developers that the system has triggered a response error and ask the relevant personnel to address it.
                output: [6, System anomaly: response error, please address urgently, -1, -1]
                Note: Argument Description for instruction code 6: arg 1: email content, arg 2: not used, always -1, arg 3: not used, always -1
                """,
            },
            {"role": "user", "content": inputdata},
        ],
    )
    # print(completion.choices[0].message)
    return completion.choices[0].message.content.strip("[]").split(",")


def classification_anomaly_openai(inputdata: str = None) -> list:
    """analyze clound run history data

     Parameters:
    - inputdata (str): system log
    return:
    - list : ['anomaly class1 : describe','anomaly class2 : describe' ]

    """
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """
                Our system primarily records five types of anomalies:
                1. CPU utilization >= 60% for 2 minutes
                2. Memory utilization >= 60% for 2 minutes
                3. Cloud run restart (startup_latency != 0)
                4. Instance count >= 2
                5. Response Fail (request count 4xx, 5xx)
                You will receive logs from users in the following format (quantity varies):
                {'yyyy-mm-dd hh:mm:ss': 'anomaly description',
                'yyyy-mm-dd hh:mm:ss': 'anomaly description'}
                Here is an example:
                {'2024-1-23 07:58:00': 'request fail. error code: 503, latencies: 0 ms',
                '2024-01-23 07:55:00': 'instance count=2 (>= 2)',
                '2024-01-23 07:56:00': 'instance count=2 (>= 2)'}
                Upon receiving user logs, you will extract error information and, for ease of program reading, must reply only in the following format without any superfluous text:
                ['error_class1: log_info (e.g., http code, cpu utilization...)','error_class2: log_info']
                Here is a complete example:
                user_input:
                {"2024-01-23 18:19:00": "instance count=2 (>= 2). other information: cpu:0.0% memory:37.63961792%     
                instance_count:2.0 request_count:http code 200:1.0 http code 404:0.0 http code 500:0.0 request_latencies:0.0 ms",}
                '"2024-01-24 08:13:00": "request fail. error code: 503. other information: cpu:0.9780119937% memory:1.48638916%     
                instance_count:1.0 request_count:http code 200:0.0 http code 404:0.0 http code 503:1.0 request_latencies:0.0 ms",}
                "2024-01-24 08:34:00": "cloud run restart at 106366.128 ms. other information: cpu:0.0% memory:2.39706421%          
   	            instance_count:1.0 request_count:http code 200:1.0 http code 404:0.0 http code 500:0.0 request_latencies:0.0 ms",}
                ['Response Fail: latencies: 0 ms, http code: 503', 'Instance Count >= 2: instance = 2','Cloud run restart: restart at 106366.128 ms']
             """,
            },
            {"role": "user", "content": inputdata},
        ],
    )
    # print(completion.choices[0].message)
    return completion.choices[0].message.content.strip("[]").split(",")


def analyze_data(inputdata: str = None) -> str:
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """
                "Our system primarily records five types of anomalies:
                CPU utilization >= 60% for 2 minutes
                Memory utilization >= 60% for 2 minutes
                Cloud run restart (startup_latency != 0)
                Instance count >= 2
                Response Failure (request count 4xx, 5xx)
                You will receive logs from users in the following format (quantity may vary):
                {'yyyy-mm-dd hh:mm:ss': 'anomaly description',
                'yyyy-mm-dd hh:mm:ss': 'anomaly description'}

                Here is an example:
                {"2024-01-23 18:19:00": "instance count=2 (>= 2). other information: cpu:0.0% memory:37.63961792%     
                instance_count:2.0 request_count:http code 200:1.0 http code 404:0.0 http code 500:0.0 request_latencies:0.0 ms",}
                '"2024-01-24 08:13:00": "request fail. error code: 503. other information: cpu:0.9780119937% memory:1.48638916%     
                instance_count:1.0 request_count:http code 200:0.0 http code 404:0.0 http code 503:1.0 request_latencies:0.0 ms",}
                "2024-01-24 08:34:00": "cloud run restart at 106366.128 ms. other information: cpu:0.0% memory:2.39706421%          
   	            instance_count:1.0 request_count:http code 200:1.0 http code 404:0.0 http code 500:0.0 request_latencies:0.0 ms",}

                Upon receiving the user's logs, you must briefly list the possible causes of the errors, focusing on analyzing system status without providing solutions. Limit the output to 80 words or less."
             """,
            },
            {"role": "user", "content": inputdata},
        ],
    )

    return completion.choices[0].message.content


def sort_log(inputdata: str = None) -> str:
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """
                "Our system primarily records five types of anomalies:
                CPU utilization >= 60% for 2 minutes
                Memory utilization >= 60% for 2 minutes
                Cloud run restart (startup_latency != 0)
                Instance count >= 2
                Response Failure (request count 4xx, 5xx)
                You will receive logs from users in the following format (quantity may vary):
                {'yyyy-mm-dd hh:mm:ss': 'anomaly description',
                'yyyy-mm-dd hh:mm:ss': 'anomaly description'}

                Here is an example:
                {"2024-01-23 18:19:00": "instance count=2 (>= 2). other information: cpu:0.0% memory:37.63961792%     
                instance_count:2.0 request_count:http code 200:1.0 http code 404:0.0 http code 500:0.0 request_latencies:0.0 ms",}
                '"2024-01-24 08:13:00": "request fail. error code: 503. other information: cpu:0.9780119937% memory:1.48638916%     
                instance_count:1.0 request_count:http code 200:0.0 http code 404:0.0 http code 503:1.0 request_latencies:0.0 ms",}
                "2024-01-24 08:34:00": "cloud run restart at 106366.128 ms. other information: cpu:0.0% memory:2.39706421%          
   	            instance_count:1.0 request_count:http code 200:1.0 http code 404:0.0 http code 500:0.0 request_latencies:0.0 ms",}

                You need to organize these logs, summarizing the times and types of anomalies briefly. Please limit the output to 100 words or less.
             """,
            },
            {"role": "user", "content": inputdata},
        ],
    )
    return completion.choices[0].message.content


def gptqa(query: str = None) -> str:
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": query},
        ],
    )

    return completion.choices[0].message.content


def analyze_vertexAI(error):
    # error must be list
    http_tmp = []  # http error code
    error_tmp = []  # others
    pattern = r"http code: (.+)"
    for message in error:
        code = re.search(pattern, message)
        # print(code)
        if code:
            http_tmp.append(code.group(1))
        else:
            error_tmp.append(message)
    set(error_tmp)
    print(error_tmp)
    print(http_tmp)

    error_list = ["CPU utilization", "Memory utilization", "Cloud run restart", "Instance count", "Response Fail"]
    error_record = np.zeros(len(error_list), dtype=int)
    for err in error:
        for err_class in range(len(error_list)):
            if error_list[err_class].lower() in err.lower():
                error_record[err_class] += 1
                break

    print(error_record)  # error type

    answer_tmp = []
    for i in range(len(error_record)):
        if error_record[i] != 0:
            if i == 4:
                for j in http_tmp:
                    query = "when will system occur " + error_list[i] + j
                    doc = total_db[3 + int(j[0])]
                    result = retrieve(doc, llm, query)
                    answer_tmp.append("problem= " + error_list[i] + j + " Ans: " + result["result"])
                    time.sleep(5)
                    print(result["result"])
            else:
                query = "when will system occur " + error_list[i]
                doc = total_db[i]
                result = retrieve(doc, llm, query)
                answer_tmp.append("problem= " + error_list[i] + " Ans: " + result["result"])
                time.sleep(5)
                print(result["result"])

    answer_tmp = list(set(answer_tmp))
    final_answer = " and ".join(answer_tmp)
    return final_answer


def resovle(inputdata: str = None) -> str:
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": """
                Users will provide you with the system's error logs and relevant analysis. Your task is to analyze this information and determine the appropriate action for addressing these issues. Your available actions are limited to:
                1.CPU upscale: Typically used when CPU usage is high or when you believe the system requires more CPU resources.
                2.Increase memory: Usually employed when memory usage is high or when you think additional memory is needed.
                3.Email system administrator: Generally used when other actions fail to resolve the problem or when you believe certain situations require notification.
                4.nothing to do : system no error exist
                Please follow this format for your response:
                action: ...  content(only email system administrator needed): ... reason: ...
                Remember, you can only execute one action at a time and your action will indeed be implemented, affecting the system. Please choose your actions carefully. Below are example outputs:
                Example 1
                action: CPU upscale. 
                reason: Currently, the CPU usage is too high and there are no other anomalies, so an increase in CPU is necessary.
                Example 2
                action: Increase memory. 
                reason: Currently, there is insufficient memory and an expansion is needed.
                Example 3
                action: Email system administrator. content:There are multiple unknown connection errors occurring, requiring contact with relevant personnel for resolution. error message:response fail 404
                reason: Currently, the authorized actions are insufficient to resolve the issue. Therefore, it is necessary to send a letter requesting the relevant personnel to address it.
                Example 4
                action: nothing to do. content: system is safity
             """,
            },
            {"role": "user", "content": inputdata},
        ],
    )

    return completion.choices[0].message.content


def real_detection(inputdata: str = None, ori_log: str = None) -> str:
    info = analyze_vertexAI(inputdata.strip("[]").split(","))
    action = resovle(f"{ori_log} \n {info}")

    return action
