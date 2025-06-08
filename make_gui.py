from tkinter import *
from tkinter import filedialog
import tkinter.ttk as ttk
import tkinter.messagebox as msg
import customtkinter as ctk

from PIL import Image, ImageTk, ImageSequence, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionPipeline
import threading
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


model_path = 'D:/backup/finetuned3'
pipe = StableDiffusionPipeline.from_pretrained(
    'stablediffusionapi/anything-v5',
    torch_dtype=torch.float16,
    safety_checker=None,
    use_auth_token=True
)
pipe.unet.load_attn_procs(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

POSITIVE_PREFIX = "(drawn by KSH drawing style : 2.0)"
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
root.geometry("1280x720")
root.configure(bg="#40434D")
root.iconbitmap("D:/backup/icon.ico")


main_frame = Frame(root, bg="#40434D")
main_frame.pack(fill="both", expand=False)

prompt_frame = Frame(main_frame, width=526, bg="#40434D", padx=30, pady=10)
prompt_frame.pack(side="left", fill="y")

new_frame = Frame(main_frame, width=3, bg="#50535B")
new_frame.pack(side="left", fill="y")

image_frame = Frame(main_frame, width=1000, bg="#40434D", padx=150, pady=10)
image_frame.pack(side="left", fill="y", expand=False)


title_img = Image.open("D:/backup/title.png")
title_img.thumbnail((200, 120))
title_img = ImageTk.PhotoImage(title_img)
title_img_label = Label(prompt_frame, image=title_img, bg="#40434D")
title_img_label.pack(pady=20)


Label(prompt_frame, text="Positive Prompt", bg="#40434D", fg="white", font=("평창 평화체 Light", 16)).pack(anchor="w")
positive_text = ctk.CTkTextbox(prompt_frame, width=426, height=180, corner_radius=10, font=("arial", 14), fg_color="#50535B", text_color="white")
positive_text.pack(fill="x", pady=(0, 20))

Label(prompt_frame, text="Negative Prompt", bg="#40434D", fg="white", font=("평창 평화체 Light", 16)).pack(anchor="w")
negative_text = ctk.CTkTextbox(prompt_frame, width=426, height=180, corner_radius=10, font=("arial", 14), fg_color="#50535B", text_color="white")
negative_text.pack(fill="x", pady=(0, 10))


loading_label = None
loading_frames = []
loading_running = False

def load_loading_gif():
    global loading_frames
    gif_path = "D:/backup/loading.gif"
    gif = Image.open(gif_path)
    
    resized_frames = []
    for frame in ImageSequence.Iterator(gif):
        resized = frame.resize((512, 512), Image.LANCZOS)
        resized_frames.append(ImageTk.PhotoImage(resized))
    
    loading_frames.extend(resized_frames)

def animate_loading(index=0):
    if not loading_running:
        return
    frame = loading_frames[index]
    loading_label.config(image=frame)
    loading_label.image = frame
    root.after(100, animate_loading, (index + 1) % len(loading_frames))

def show_loading():
    global loading_label, loading_running
    if loading_label is None:
        loading_label = Label(image_frame, bg="#40434D")
        loading_label.pack(pady=0)
    loading_running = True
    animate_loading()

def hide_loading():
    global loading_running
    loading_running = False
    if loading_label:
        loading_label.config(image="")
        loading_label.image = None

load_loading_gif()


status_label = Label(image_frame, text="", font=("평창 평화체 Light", 20), bg="#40434D", fg="white")
status_label.pack(pady=10)

image_label = Label(image_frame, bg="#40434D")
image_label.pack()

save_button = None
generated_image = None


def generate_image(prompt, neg_prompt):
    global generated_image, save_button

    status_label.config(text="이미지 생성 중...")
    image_label.config(image="")
    if save_button:
        save_button.destroy()

    show_loading()

    
    image = pipe(prompt, negative_prompt=neg_prompt,
                 num_inference_steps=30, guidance_scale=7.5).images[0]

    hide_loading()

    status_label.config(text="이미지 생성 완료!")
    generated_image = image

    resized = image.resize((512, 512))
    tk_image = ImageTk.PhotoImage(resized)
    image_label.config(image=tk_image)
    image_label.image = tk_image

    
    button_bg = Image.open("D:/backup/button.png").resize((180, 50), Image.LANCZOS)
    save_img = ImageTk.PhotoImage(draw_button_with_text(button_bg, "Save"))
    save_button = Button(image_frame, image=save_img, command=save_image, borderwidth=0, bg="#40434D", activebackground="#40434D")
    save_button.image = save_img
    save_button.pack(pady=0)


def draw_button_with_text(image, text):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Users/USER/AppData/Local/Microsoft/Windows/Fonts/PyeongChangPeace-Light.ttf", 25)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    image_center_x = image.width / 2
    image_center_y = image.height / 2

    text_x = image_center_x - text_width / 2
    text_y = (image_center_y - text_height / 2) - 5

    draw.text((text_x, text_y), text, font=font, fill="white")
    return image



def save_image():
    if generated_image:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            initialfile="character.png"
        )
        if file_path:
            generated_image.save(file_path)
            msg.showinfo("저장 완료", f"이미지가 저장되었습니다")


def start_generation():
    answer = msg.askyesno("확인", "이미지 생성을 시작하시겠습니까?")
    if answer:
        pos = positive_text.get("1.0", "end-1c").strip()
        neg = negative_text.get("1.0", "end-1c").strip()
        prompt = POSITIVE_PREFIX + pos
        neg_prompt = NEGATIVE_PREFIX + neg
        show_loading()
        threading.Thread(target=generate_image, args=(prompt, neg_prompt)).start()


button_img_raw = Image.open("D:/backup/button.png").resize((180, 50), Image.LANCZOS)
button_img = ImageTk.PhotoImage(draw_button_with_text(button_img_raw, "Generate"))
generate_button = Button(prompt_frame, image=button_img, command=start_generation, borderwidth=0, bg="#40434D", activebackground="#40434D")
generate_button.image = button_img
generate_button.pack(anchor="e", pady=10)


root.mainloop()

