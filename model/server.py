import asyncio,json
from time import sleep,time
import re,os
import hashlib  
import threading
from datetime import datetime

#######################################
#
# 任务池：TASK_POOL,形式[{"order_id":int, "hash_path":srt}]，长度为每日最大请求数MAX_REQUEST_TIMES_DAY
# {'req_type': 'ask', 'order_id': 799101}重置最大请求代码
#######################################

def getToday():
    return datetime.now().strftime("%d")

#----------------------
# gloal key var
MAX_REQUEST_TIMES_DAY_CONSTANT = 100
MAX_REQUEST_TIMES_DAY = MAX_REQUEST_TIMES_DAY_CONSTANT
IS_OPEN = True
USER_ROOT_DIR = "/var/www/html/log" #"E:\\code\\python\\5.异步服务器\\log" #"/var/www/html/log"
TODAY = getToday()

#----------------------
# md5 哈希计算函数
def calculate_md5(input_string):  
    md5_hash = hashlib.md5()  
    md5_hash.update(input_string.encode("UTF8"))    
    return md5_hash.hexdigest()  

def getUserHashName(seqs):
    string = "%s-%s"%(time(), seqs)
    user_hash_name = calculate_md5(string)
    return user_hash_name

def getUserHashDir(seqs):
    global USER_ROOT_DIR
    # seqs：提交的序列
    string = "%s-%s"%(time(), seqs)
    user_hash_name = calculate_md5(string)
    user_hash_dir = "%s/%s"%(USER_ROOT_DIR,user_hash_name)
    user_results_dir  = "%s/results/"%(user_hash_dir)
    user_plt_results_dir = "%s/plt_results/"%(user_hash_dir)
    user_file_path = "%s/user.csv"%(user_hash_dir)
    # 
    createUserFiles(user_hash_dir, user_results_dir, user_plt_results_dir, seqs, user_file_path)
    return user_hash_name, user_hash_dir, user_results_dir, user_plt_results_dir,user_file_path

def selectMkdir(path):
    if os.path.exists(path):
        return
    command = "mkdir %s"%(path)
    print(command)
    os.system(command)

def createUserFiles(user_hash_dir, user_results_dir, user_plt_results_dir, seq, user_file_path):
    # create dir
    selectMkdir(user_hash_dir)
    selectMkdir(user_results_dir)
    selectMkdir(user_plt_results_dir)
    # user.csv
    f = open(user_file_path,"w")
    f.write("Name,Sequence\n")
    f.write("seq1,%s\n"%(seq))
    f.close()

def execCOMDEL2_Pred(user_hash_name, seqs):
    global USER_ROOT_DIR
    # 生成必要文件
    user_hash_dir = "%s/%s"%(USER_ROOT_DIR,user_hash_name)
    user_results_dir  = "%s/results/"%(user_hash_dir)
    user_plt_results_dir = "%s/plt_results/"%(user_hash_dir)
    user_file_path = "%s/user.csv"%(user_hash_dir)
    createUserFiles(user_hash_dir, user_results_dir, user_plt_results_dir, seqs, user_file_path)
    #-----------------------
    # 预测抗菌肽
    COMDEL2_work_dir = "/home/lw/COMDEL2/predict"
    COMDEL1_work_dir = "/home/lw/COMDEL2/COMDEL"
    os.chdir(COMDEL1_work_dir)
    command = "python3 COMDEL2_AMP.py -cuda -infile %s -results %s -plt_results %s"%(user_file_path, user_results_dir, user_plt_results_dir)
    os.system(command)
    # 预测其他
    os.chdir(COMDEL2_work_dir)
    command = "python3 predict.py --infile %s --result_path %s --plt_path %s"%(user_file_path, user_results_dir, user_plt_results_dir)
    os.system(command)
    # --------------------------
    # R语言运行获取热图
    heat_map_in_file = "%s/Combined_predict.csv"%(user_results_dir)
    heat_map_out_file = "%s/predseq_circular_heatmap.png"%(user_plt_results_dir)
    command = "Rscript pred_seq_circular.R %s %s"%(heat_map_in_file, heat_map_out_file)
    os.system(command)     
    #---------------------------
    # 压缩文件
    zip_path = "%s/%s"%(USER_ROOT_DIR, user_hash_name)
    os.chdir(zip_path)
    command = "zip -r results.zip ./"
    os.system(command)
    sleep(2)

class COMDEL2_EXEC(threading.Thread):
    def __init__(self,name):
        # # task 是Task类
        super().__init__(name=name)
    
    def run(self):
        global TODAY, MAX_REQUEST_TIMES_DAY
        while True:
            cur_day = getToday()
            if cur_day != TODAY:
                TODAY = cur_day
                MAX_REQUEST_TIMES_DAY = MAX_REQUEST_TIMES_DAY_CONSTANT
                TASK_POOL.clear()
            for task in TASK_POOL.pool:
                status = task.task["status"]
                if status=="done":
                    continue
                hash_path = task.task["hash_path"]
                seqs = task.task["seqs"]
                task.task["status"]="running"
                execCOMDEL2_Pred(hash_path, seqs)
                task.task["status"]="done"

#----------------------
# task
class Task:
    def __init__(self,hash_path,status,seqs):
        # status: wait done runing
        self.task = {"hash_path":hash_path,"status":status, "seqs":seqs}

#----------------------
# TASK_POOL class
class TaskPool:
    def __init__(self):
        self.pool = []

    def isExMaxRequestTimes(self):
        global MAX_REQUEST_TIMES_DAY
        # "是否超过最大请求数"
        if MAX_REQUEST_TIMES_DAY==0:
            return True
        return False
    
    def add(self, task):
        global MAX_REQUEST_TIMES_DAY
        if not self.isExMaxRequestTimes():
            MAX_REQUEST_TIMES_DAY -= 1
            self.pool.append(task)
            order_num = len(self.pool)
            return {"order_id":order_num-1,"status":"add_success","success":1,"mes":"Success add in task line. Order in %dth"%order_num}
        return {"status":"error","success":0,"mes":"Available task is run out in today"}
    
    def clear(self):
        global USER_ROOT_DIR
        # while len(self.pool)>0:
        #     self.pool.pop(0)
        # 清空用户文件
        command = "rm -rf %s/*"%(USER_ROOT_DIR)
        os.system(command)
        # 清空任务池
        self.pool = []
    
    def doneMes(self,user_hash_name,seqs):
        url_img_prefix =  "/log/%s/plt_results"%(user_hash_name)
        zip_url = "/log/%s/results.zip"%(user_hash_name)
        img_list = [
            "ACP_predictions.png","ADP_predictions.png",
            "AGP_predictions.png","AHP_predictions.png",
            "AIP_predictions.png","AMP_predictions.png",
            "BiP_predictions.png","CPP_predictions.png",
            "DDP_predictions.png","DeP_predictions.png",
            "HeP_predictions.png","NuP_predictions.png",
            "UmP_predictions.png","predseq_circular_heatmap.png",
        ]
        imgs_url = []
        for img_name in img_list:
            img_url = "%s/%s"%(url_img_prefix,img_name)
            imgs_url.append(img_url)
        res_results = {
            "status":"done",
            "success":1,
            "imgs":imgs_url,
            "seqs":seqs,
            "zip_url":zip_url
        }
        return res_results
    
    def ask(self,data):
        global MAX_REQUEST_TIMES_DAY, IS_OPEN, TASK_POOL
        # 问询{"req_type":"ask","order_id":int}
        if "order_id" not in data:
            return {"status":"error","success":0, "mes":"Ask parmas is invalid no order_id"}
        try:
            ask_i = int(data["order_id"])
        except:
            return {"status":"error","success":0, "mes":"Ask parmas is invalid order_id not num"}
        if ask_i>len(self.pool)-1:
            if ask_i==799101:
                MAX_REQUEST_TIMES_DAY = MAX_REQUEST_TIMES_DAY_CONSTANT
                return {"status":"restart","success":1, "mes":"Reset requests num"}
            if ask_i==799102:
                IS_OPEN = False
                return {"status":"COMDEL2 close","success":1, "mes":"Close Comdel2_pred"}
            if ask_i==799103:
                IS_OPEN = True
                return {"status":"COMDEL2 start","success":1, "mes":"Start Comdel2_pred"}
            if ask_i==799104:
                TASK_POOL.clear()
                return {"status":"clear task pool","success":1, "mes":"clear task pool done"}
            return {"status":"error","success":0, "mes":"Exceed max num of task pool"}
        # wait done running
        task = self.pool[ask_i].task
        status = task["status"]
        if status=="wait":
            cur_i = ask_i
            while cur_i>0:
                cur_i -= 1
                cur_task = self.pool[cur_i].task
                if cur_task["status"]=="running":
                    break
            return {"status":"wait","success":1, "mes":"Please wait above you have %d task"%(ask_i-cur_i)}
        if status=="running":
            return {"status":"running","success":1, "mes":"Please wait your task is running"}
        if status=="done":
            mes = self.doneMes(task["hash_path"],task["seqs"])
            return mes
        return {"status":"error","success":0, "mes":"Unknow task status"}

#----------------------
# json响应头体
class jsonResponseHeadBody:
    def __init__(self, dict_data):
        res_head_dict = [
            "HTTP/1.1 200 OK",
            "Content-Type: application/json; charset=UTF-8",
            "",""
        ]
        res_head = "\r\n".join(res_head_dict)
        res_body = json.dumps(dict_data)
        self.res = res_head + res_body
    
    def getResStr(self):
        return self.res.encode("UTF8")

#--------------------------
# 异步服务器类
class AsynServer:
    def __init__(self, host="localhost", port=8899):
        self.host = host
        self.port = port 
        self.req_dict = None

    def parsePostData(self,req_level_1):
        post_data_arr = req_level_1[0].split(" ")[1].split("?")
        table = {}
        if len(post_data_arr)>1:
            post_data_str = post_data_arr[1]
        else:
            post_data_str = req_level_1[-1]
        if not post_data_str:
            return {}
        arr = post_data_str.split("&")
        for d in arr:
            p_k = d.split("=")[0]
            p_v = d.split("=")[1]
            table[p_k] = p_v
        return table

    def parseReqHead(self, data):
        # 解析post请求头返回头部数据和体数据
        data = data.decode("UTF8")
        arr_level_1 = data.split("\r\n")
        req_head = {}
        req_type = arr_level_1[0].split(" ")[0].upper()
        # get post data
        post_data = self.parsePostData(arr_level_1)
        # get head dict
        post_url = arr_level_1[0].split(" ")[1].split("?")[0]
        for i in range(1, len(arr_level_1)):
            if not arr_level_1[i]:break
            arr_level_2 = arr_level_1[i].split(": ")
            req_head[arr_level_2[0]] = arr_level_2[1]
        result = {"head":req_head,
                  "body":post_data,
                  "req_type":req_type,
                  "post_url":post_url
                }
        self.req_dict = result 
        print("请求解析结果", result) 

    def isPostReq(self):
        # 是否是POST请求
        if self.req_dict["req_type"]!="POST":
            return False
        return True
    
    def isPostEmpty(self):
        # 是否POST上传数据为空
        if self.req_dict["body"]:
            return False
        return True
    
    async def sendMes(self,writer,mes_dict):
        json_res = jsonResponseHeadBody(mes_dict)
        json_res_cont = json_res.getResStr()
        writer.write(json_res_cont)
        await writer.drain()
        writer.close()
        # json_res:jsonResponseHeadBody
        
    
    def isPostDataValid(self):
        # 是否POST上传数据合法
        # 两种类型
        # 1. 提交的数据是否合法：氨基酸残基，大小写，长度限制
        # 2. 已经提交数据的用户每轮的问询请求数据是否合法
        pass
    
    async def validPostData(self,writer):
        # 非POST数据？POST数据内容不正确？正确处理等分支流程
        # 获取请求体数据
        # 氨基酸数据处理
        # 提交参数不正确
        if "seqs" not in self.req_dict["body"]:
            await self.sendMes(writer, {"status":"error","success":0,"mes":"Input seq is empty no parmas"})
            return
        # 提交为空
        if not self.req_dict["body"]["seqs"]:
            await self.sendMes(writer, {"status":"error","success":0,"mes":"Input seq is empty"})
            return
        # 提交非残基
        aa_seq = re.sub("\s","",self.req_dict["body"]["seqs"])
        aa_seq = aa_seq.upper()
        valid_aa_reg = re.search("[^AFCUDNEQGHLIKOMPRSTVWY]", aa_seq)
        if valid_aa_reg:
            await self.sendMes(writer, {"status":"error","success":0,"mes":"Illegal amino acid character"})
            return
        if len(aa_seq)>2000:
            await self.sendMes(writer, {"status":"error","success":0,"mes":"Length of amino acid must less than 2000"})
            return
        self.req_dict["body"]["seqs"] = aa_seq
        return True

    async def handleReq(self, reader, writer):
        global IS_OPEN
        # 处理客户端请求
        # 提交和问询请求
        # 提交{"req_type":"submit","seqs":aa}
        # 问询{"req_type":"ask","order_id":int}
        # 数据初始化
        req_raw_data = await reader.read(3000)
        try:
            self.parseReqHead(req_raw_data)
        except:
            # 错误信息返回，无效数据
            await self.sendMes(writer, {"status":"error","success":0,"mes":"Invalid data"})
            return
        # 数据格式是否合法
        if len(self.req_dict["body"])!=2:
            await self.sendMes(writer, {"status":"error","success":0,"mes":"Invalid data format params must be two"})
            return
        if "req_type" not in self.req_dict["body"]:
            await self.sendMes(writer, {"status":"error","success":0,"mes":"Invalid data format no requests type"})
            return
        if self.req_dict["body"]["req_type"] not in ["ask","submit"]:
            await self.sendMes(writer, {"status":"error","success":0,"mes":"Invalid data format requests type error"})
            return
        #---------------------------
        # 问询和提交操作
        if self.req_dict["body"]["req_type"]=="ask":
            # 问询操作
            mes = TASK_POOL.ask(self.req_dict["body"])
            await self.sendMes(writer,mes)
        # 提交操作
        elif self.req_dict["body"]["req_type"]=="submit":
            if not IS_OPEN:
                await self.sendMes(writer, {"status":"error","success":0,"mes":"COMDEL2 pred is close"})
                return
            if await self.validPostData(writer):
                # res = execCOMDEL2_Pred(self.req_dict["body"]["seqs"])
                # await self.sendMes(writer,res)
                seqs = self.req_dict["body"]["seqs"]
                user_hash_name = getUserHashName(seqs)
                task = Task(user_hash_name,"wait",seqs)
                mes = TASK_POOL.add(task)
                await self.sendMes(writer,mes)

    async def run(self):
        server = await asyncio.start_server(self.handleReq, self.host, self.port)
        addr = server.sockets[0].getsockname()  
        print(f'Serving on {addr}')  
        async with server:  
            await server.serve_forever()  

# Python 3.7+  
if __name__ == "__main__":  
    # asyncio.run(main())
    TASK_POOL = TaskPool()
    comdel2_exec = COMDEL2_EXEC("comdel2")
    comdel2_exec.daemon = True
    comdel2_exec.start()
    ASY_SERVER = AsynServer()
    asyncio.run(ASY_SERVER.run())
