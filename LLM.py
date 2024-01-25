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


def embedding_all_doc():
    url_cpu = "CPU utilization.pdf"
    db_cpu = doc2vector(url_cpu)
    time.sleep(5)

    url_Memory = "Memory utilization.pdf"
    db_Memory = doc2vector(url_Memory)
    time.sleep(5)

    url_Cloud = "Cloud run restart.pdf"
    db_Cloud = doc2vector(url_Cloud)
    time.sleep(5)

    url_Instance = "Instance count.pdf"
    db_Instance = doc2vector(url_Instance)
    time.sleep(5)

    url = "http_1_3.pdf"
    db = doc2vector(url)
    time.sleep(30)

    url2 = "http_4.pdf"
    db2 = doc2vector(url2)
    time.sleep(30)

    url3 = "http_5.pdf"
    db3 = doc2vector(url3)

    total_db = [db_cpu, db_Memory, db_Cloud, db_Instance, db, db, db, db2, db3]
    return total_db


total_db = embedding_all_doc()


def get_function_openai(inputdata: str = "Hello") -> list:
    """
    �鍂靘��𦻖�𤣰��滨垢雿輻鍂����頛詨�伐����ế�𪃾�糓撅祆䲰�𪑛銝�蝔桅�𧼮��
     Parameters:
    - inputdata (str): 雿輻鍂��頛詨��
    return:
    - list: ['class','arg1','arg2','arg3']

    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
                Our system has the following functions:
                1. GPT Q&A
                2. Analysis of historical logs for the past n days
                3. CPU upscaling
                4. Memory upscaling
                5. Real-time data monitoring
                You will receive various inputs from users, and your task is to determine which category the task belongs to and output it in the format ['function code', 'arg 1', 'arg 2', 'arg 3']:
                Here are some examples:
                User input: 'How are you?'
                Output: [1, How are you?, -1, -1]
                Different functions require different parameters. You must determine what to fill in for each parameter. Additionally, avoid generating any extra text for ease of programming:
                Example 1:
                User input: 'I want to understand HTTP CODE'
                Output: [1,I want to understand HTTP CODE, -1, -1]
                Example 2:
                User input: 'Check if there have been any anomalies in my system in the past 5 days, 30 hours, and 70 minutes'
                Output: [2, 5, 30, 70]
                Note: No need to convert time units.
                Example 3:
                User input: 'Check if there have been any anomalies in my system in the past 30 hours'
                Output: [2, -1, 30, -1]
                Example 4:
                User input: 'Please find the latest information on memory from the past few days.'
                Output: [2, 1, -1, -1]
                Note: If no specific time is designated, the default is 1 day.
                Example 5:
                user_input: 'My CPU space is not enough, please increase the CPU capacity '
                output: [3, add, -1, -1]
                Note: Since it's a request to increase the CPU, arg1 = add
                Example 6:
                user_input: 'I have too much CPU space, reduce 2 CPUs for me'
                output: [3, sub, -1, -1]
                Note: Since it's a request to reduce the CPU, arg1 = sub. Additionally, the number of CPUs to be used is not specified, so no need to fill in 2.
                Example 7:
                user_input: 'My memory space is not enough, please increase the memory capacity by 128m'
                output: [4, add, -1, -1]
                Note: Memory capacity cannot be specified, only a request to increase is needed, so no need to fill in the parameter 128.
                Example 8:
                user_input: 'My memory space is overflowing, please reduce the memory capacity'
                output: [4, sub, -1, -1]
                Example 9:
                user_input: 'I need you to help me manage my system'
                output: [5, start, -1, -1]
                Note: since start manage system,arg1=start
                Example 10:
                user_input: 'You can stop helping me manage the system now'
                output: [5, stop, -1, -1]"
                Note: since start manage system,arg1=stop
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
        model="gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
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

                Upon receiving these logs from users, you will analyze them to summarize the past state of the system and the possible reasons for these anomalies. Remember to focus on inferring the state of the system rather than providing solutions."
             """,
            },
            {"role": "user", "content": inputdata},
        ],
    )

    return completion.choices[0].message.content


def gptqa(query: str = None) -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """
                We will provide you with the error log and relevant error analysis of our system. Your task is to analyze this information and determine the appropriate actions to resolve these issues. The actions you can take include, but are not limited to:
                1.CPU upscale
                2.Memory upscale
                3.Sending emails to notify relevant personnel and provide your opinion
                4.nothing to do
                5.Any other suggestions
                Please respond to the user with your chosen action in 20 words or less (excluding email content, if that's your choice). 
                Remember, you can only take one action at a time, and your decision will be implemented and affect the system, so choose carefully.
             """,
            },
            {"role": "user", "content": inputdata},
        ],
    )

    return completion.choices[0].message.content


def real_detection(inputdata: str = None) -> str:
    info = analyze_vertexAI(inputdata.strip("[]").split(","))
    action = resovle(f"{inputdata} \n {info}")

    return action