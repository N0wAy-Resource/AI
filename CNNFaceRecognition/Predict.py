import numpy as np
import cv2
import os
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示WARNING和ERROR
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

#初始化模型变量
model = None
class_reference_images= None
class_names = None
test_image_path= None

# 预测和显示函数
def predict_and_display( model, class_ref_images, img_size=(128, 128)):
    # 预处理输入图像
    img = cv2.imdecode(
            np.fromfile(test_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("Error: Image not found")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img_array = np.expand_dims(img, axis=0) / 255.0

    # 进行预测
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    # if(confidence<0.5):
    #     tk.messagebox.showerror("提示", "匹配失败")
    #     return
    # 获取参考图像
    ref_image = class_ref_images.get(predicted_class, np.zeros((*img_size, 3)))
    # 转换numpy数组为PIL图像
    ref_image_pil = Image.fromarray((ref_image * 255).astype('uint8'))
    # 显示结果
    ref_image_pil=ref_image_pil.resize((200,250))
    show_image(ref_image_pil, result_canvas)
    # 更新置信率文本
    confidence_label.config(text=f"预测置信度: {confidence:.2%}")

def loadmodel():
    global model, class_reference_images,class_names
    model_path = "face_recognition_cnn.keras"
    model = keras.api.saving.load_model(model_path)
    metadata = np.load(model_path.replace('.keras', '_metadata.npz'), allow_pickle=True)
    class_names = metadata['class_names'].tolist()
    class_reference_images = metadata['class_reference_images'].item()
    tk.messagebox.showinfo("提示", "模型加载成功")

def select_image():
    global test_image_path
    path = filedialog.askopenfilename(filetypes=[("JPEG文件", "*.jpg")])
    if path:
        img = Image.open(path)
        show_image(img, test_canvas)
        test_image_path = os.path.normpath(path)
        print("[DEBUG] 当前路径:", test_image_path)  # 调试输出
def match_face():
    global test_image_path,model,class_reference_images,class_names
    if model==None:
        tk.messagebox.showerror("错误", "请先加载模型")
        return
    predict_and_display( model, class_reference_images)

def show_image(img, canvas):
    img=img.resize((200, 250))
    photo = ImageTk.PhotoImage(img)
    canvas.image = photo  # 保持引用
    canvas.create_image(0, 0, anchor='nw', image=photo)

# 创建主窗口
root = tk.Tk()
root.title("CNN人脸识别系统")

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
test_canvas = tk.Canvas(frame, width=200, height=250)
test_canvas.grid(row=1, column=0)

# 添加置信率文本
confidence_label = ttk.Label(frame, text="预测置信度: 0.00%")
confidence_label.grid(row=1, column=1, padx=10)

# 匹配结果显示区域
result_img_label = ttk.Label(frame, text="匹配结果")
result_img_label.grid(row=0, column=2, padx=5)
result_canvas = tk.Canvas(frame, width=200, height=250)
result_canvas.grid(row=1, column=2)

btn_frame = ttk.Frame(frame)
btn_frame.grid(row=2, column=0, columnspan=3, pady=10)

select_btn = ttk.Button(btn_frame, text="选择测试图片", command=select_image)
select_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

match_btn = ttk.Button(btn_frame, text="开始匹配", command=match_face)
match_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

load_btn = ttk.Button(btn_frame, text="加载模型", command=loadmodel)
load_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

root.mainloop()