import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

# 在代码开头添加模型路径定义
MODEL_FILES = {
    'mean': 'mean_face.npy',
    'eigenface': 'eigenface.npy',
    'train_proj': 'eigen_train_sample.npy',
    'image_paths': 'image_paths.npy'
}

# 初始化模型变量
meanFaceMat = None
eigenface = None
eigen_train_sample = None
image_paths = None

# 程序启动时自动尝试加载模型
def load_model():
    global meanFaceMat, eigenface, eigen_train_sample,image_paths
    try:
        if all(os.path.exists(f) for f in MODEL_FILES.values()):
            meanFaceMat = np.load(MODEL_FILES['mean'])
            eigenface = np.load(MODEL_FILES['eigenface'])
            eigen_train_sample = np.load(MODEL_FILES['train_proj'])
            image_paths = np.load(MODEL_FILES['image_paths'], allow_pickle=True).tolist()
            return True
        return False
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return False

load_model()  # 启动时加载

# 创建主窗口
root = tk.Tk()
root.title("PCA人脸识别系统")

# 导入人脸数据
test_image_path = None
classifier_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(classifier_path):
    print("Error: Classifier file not found!")
    exit(1)
faceCascade = cv2.CascadeClassifier(classifier_path)

# 在GUI界面添加状态标签和训练按钮
status_frame = ttk.Frame(root)
status_frame.grid(row=1, column=0, sticky="ew")
root.status_label = ttk.Label(status_frame, text="程序就绪")
root.status_label.pack()

# GUI组件
frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0)

# 测试图片显示区域
test_img_label = ttk.Label(frame, text="测试图像")
test_img_label.grid(row=0, column=0, padx=5)
test_canvas = tk.Canvas(frame, width=200, height=200)
test_canvas.grid(row=1, column=0)

# 匹配结果显示区域
result_img_label = ttk.Label(frame, text="匹配结果")
result_img_label.grid(row=0, column=1, padx=5)
result_canvas = tk.Canvas(frame, width=200, height=200)
result_canvas.grid(row=1, column=1)

# 功能按钮
def train_model():
    global images, image_paths, meanFaceMat, eigenface, eigen_train_sample

    # 清空旧数据
    images = []
    image_paths = []

    # 训练数据采集
    for i in range(1, 43):
        folder_path = os.path.normpath(f'trainface/t{i}')  # 标准化路径
        if not os.path.exists(folder_path):
            continue

        # 添加进度提示
        root.status_label.config(text=f"正在处理 {folder_path}...")
        root.update_idletasks()

        # 处理每张图片
        for img_file in [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]:
            img_path = os.path.join(folder_path, img_file)
            image = cv2.imread(img_path, 0)
            if image is None:
                continue

            # 人脸检测与处理
            image = cv2.equalizeHist(image)
            faces = faceCascade.detectMultiScale(image, 1.1, 3)
            for (x, y, w, h) in faces:
                face_roi = cv2.resize(image[y:y + h, x:x + w], (64, 64))
                images.append(face_roi.flatten())
                image_paths.append(img_path)

    # 转换为矩阵
    trainFaceMat = np.array(images,dtype=np.float32)
    meanFaceMat = np.mean(trainFaceMat, axis=0)
    normTrainFaceMat = trainFaceMat - meanFaceMat

    # 计算协方差矩阵
    covariance = np.cov(normTrainFaceMat)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # 选择前K个特征向量
    K = 140
    sorted_indices = np.argsort(eigenvalues)
    topk_evecs = eigenvectors[:, sorted_indices[:-K-1:-1]]

    # 计算特征脸空间
    eigenface = np.dot(normTrainFaceMat.T,topk_evecs)

    # 计算训练样本在特征脸空间中的投影
    eigen_train_sample = np.dot(normTrainFaceMat, eigenface)

    # 保存模型
    np.save(MODEL_FILES['mean'], meanFaceMat)
    np.save(MODEL_FILES['eigenface'], eigenface)
    np.save(MODEL_FILES['train_proj'], eigen_train_sample)
    np.save(MODEL_FILES['image_paths'], np.array(image_paths, dtype=object))

    # 训练完成提示
    root.status_label.config(text="训练完成")
    tk.messagebox.showinfo("系统提示", "模型训练完成并保存！")

def select_image():
    global test_image_path
    path = filedialog.askopenfilename(filetypes=[("JPEG文件", "*.jpg")])
    if path:
        show_image(path, test_canvas)
        test_image_path = os.path.normpath(path)
        print("[DEBUG] 当前路径:", test_image_path)  # 调试输出


def match_face():
    # 检查模型是否加载
    if any(v is None for v in [meanFaceMat, eigenface, eigen_train_sample]):
        tk.messagebox.showwarning("操作错误", "请先进行模型训练！")
        return
    global test_image_path
    try:
        if not test_image_path:  # 增加空值检查
            tk.messagebox.showwarning("警告", "请先选择测试图片")
            return
        if not os.path.exists(test_image_path):
            tk.messagebox.showerror("错误", f"文件不存在: {test_image_path}")
            return
        test_image = cv2.imdecode(
            np.fromfile(test_image_path, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        if test_image is None:  # 新增空值检查
            tk.messagebox.showerror("错误", f"图片读取失败，请检查文件格式或内容")
            return
        # 导入测试样本
        images = []
        test_image = cv2.equalizeHist(test_image)
        faces = faceCascade.detectMultiScale(test_image, 1.1, 3)
        if len(faces) == 0:
            tk.messagebox.showerror("错误", "未检测到有效人脸")
            return
        elif len(faces) > 1:  # 新增多脸处理逻辑
                # 取面积最大的人脸
                areas = [(x, y, w, h) for (x, y, w, h) in faces]
                main_face = max(areas, key=lambda item: item[2] * item[3])
                faces = [main_face]
        for (x, y, w, h) in faces:
            cutResize = cv2.resize(test_image[y:y + h, x:x + w], (64, 64), interpolation=cv2.INTER_CUBIC)
            images.append(cutResize.flatten())

        # 规格化测试样本
        testFaceMat = np.array(images, dtype=np.float32)
        normTestFaceMat = testFaceMat - meanFaceMat

        # 投影到特征脸空间
        eigen_test_sample = np.dot(normTestFaceMat, eigenface)
        # 计算欧氏距离
        distances = np.linalg.norm(eigen_train_sample - eigen_test_sample, axis=1)
        min_index = np.argmin(distances)
        # 显示匹配结果
        matched_path = image_paths[min_index]
        show_image(matched_path, result_canvas)
    except Exception as e:
        tk.messagebox.showerror("错误", str(e))


def show_image(img_path, canvas):
    img = Image.open(img_path)
    img.thumbnail((200, 200))
    photo = ImageTk.PhotoImage(img)
    canvas.image = photo  # 保持引用
    canvas.create_image(0, 0, anchor='nw', image=photo)



btn_frame = ttk.Frame(frame)
btn_frame.grid(row=2, column=0, columnspan=2, pady=10)

select_btn = ttk.Button(btn_frame, text="选择测试图片", command=select_image)
select_btn.pack(side=tk.LEFT, padx=5)

match_btn = ttk.Button(btn_frame, text="开始匹配", command=match_face)
match_btn.pack(side=tk.LEFT, padx=5)

# 在按钮区添加训练按钮
train_btn = ttk.Button(btn_frame, text="训练模型", command=train_model)
train_btn.pack(side=tk.LEFT, padx=5)
root.mainloop()