from tkinter import *
import tkinter.ttk as ttk
import tkinter.messagebox as msg

from PIL import Image, ImageTk
import torch
from diffusers import StableDiffusionPipeline
import threading
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model_path = 'HLife15/kshdrawing'
pipe = StableDiffusionPipeline.from_pretrained(
    'stablediffusionapi/anything-v5',
    torch_dtype=torch.float16,
    safety_checker = None,
    use_auth_token=True
)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

POSITIVE_PREFIX = "(drawn by KSH drawing style : 1.5)"
NEGATIVE_PREFIX = """FastNegativeV2,(bad-artist:1.0), 
(worst quality, low quality:1.4), (bad_prompt_version2:0.8),
bad-hands-5,lowres, bad anatomy, bad hands, ((text)), (watermark),
error, missing fingers, extra digit, fewer digits, cropped,
worst quality, low quality, normal quality, ((username)), blurry,
(extra limbs), bad-artist-anime, badhandv4, EasyNegative,
ng_deepnegative_v1_75t, verybadimagenegative_v1.3, BadDream,
(three hands:1.1),(three legs:1.1),(more than two hands:1.4),
(more than two legs,:1.2),badhandv4,EasyNegative,ng_deepnegative_v1_75t,
verybadimagenegative_v1.3,(worst quality, low quality:1.4),text,words,logo,watermark,"""

root = Tk()
root.title("KSH Drawing AI")
root.geometry("1000x650")
root.iconbitmap("D:/backup/icon.ico")

title_frame = Frame(root, bg="#2b2b2b", height=50)
title_frame.pack(side="top", fill="x")

title_label = Label(title_frame, text="KSH Drawing AI", font=("Arial", 20, "bold"), fg="white", bg="#2b2b2b")
title_label.pack(pady=10)

main_frame = Frame(root)
main_frame.pack(fill="both", expand=True)

prompt_frame = Frame(main_frame, width=400, padx=10, pady=10)
prompt_frame.pack(side="left", fill="y")

image_frame = Frame(main_frame, width=600, padx=10, pady=10)
image_frame.pack(side="right", fill="both", expand=True)

Label(prompt_frame, text="Positive Prompt").pack(anchor="w")
positive_text = Text(prompt_frame, height=10, wrap="word")
positive_text.pack(fill="x", pady=(0, 10))

Label(prompt_frame, text="Negative Prompt").pack(anchor="w")
negative_text = Text(prompt_frame, height=10, wrap="word")
negative_text.pack(fill="x", pady=(0, 10))

def start_generation():
    answer = msg.askyesno("확인", "이미지 생성을 시작하시겠습니까?")
    if answer:
        pos = positive_text.get("1.0", "end-1c").strip()
        neg = negative_text.get("1.0", "end-1c").strip()
        prompt = POSITIVE_PREFIX + pos
        neg_prompt = NEGATIVE_PREFIX + neg
        threading.Thread(target=generate_image, args=(prompt, neg_prompt)).start()

generate_button = Button(prompt_frame, text="이미지 생성", command=start_generation)
generate_button.pack(pady=10)

status_label = Label(image_frame, text="", font=("Arial", 14))
status_label.pack()

progress = ttk.Progressbar(image_frame, length=400, mode='determinate', maximum=100)
progress["value"] = 0
progress.pack_forget()

image_label = Label(image_frame)
image_label.pack()

save_button = None
generated_image = None

def update_progress_simulated(step=0):
    if step <= 100:
        progress["value"] = step
        root.after(50, update_progress_simulated, step + 1.7)

def generate_image(prompt, neg_prompt):
    global generated_image, save_button

    status_label.config(text="이미지 생성 중...")
    image_label.config(image="")
    if save_button:
        save_button.destroy()

    progress["value"]
    progress.pack(pady=10)
    update_progress_simulated()

    image = pipe(prompt, negative_prompt=neg_prompt,
                 num_inference_steps=30, guidance_scale=7.5).images[0]

    progress.stop()
    progress.pack_forget()

    status_label.config(text="이미지 생성 완료!")

    generated_image = image

    resized = image.resize((400, 400))
    tk_image = ImageTk.PhotoImage(resized)
    image_label.config(image=tk_image)
    image_label.image = tk_image  # 참조 유지

    save_button = Button(image_frame, text="저장", command=save_image)
    save_button.pack(pady=10)

def save_image():
    if generated_image:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            initialfile="character.png"
        )
        if file_path:
            generated_image.save(file_path)
            messagebox.showinfo("저장 완료", f"이미지가 저장되었습니다:\n{file_path}")

root.mainloop()
