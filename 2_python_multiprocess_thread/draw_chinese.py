import cv2
import numpy as np
from PIL import ImageFont,ImageDraw,Image

def draw(img:np.ndarray,x:int,y:int,chinese:str,i,j): 
    #导入字体文件
    fontpath = "./fonts/simsun.ttc"
    #设置字体的颜色
    b,g,r,a = 0,255,0,0
    #设置字体大小
    font = ImageFont.truetype(fontpath,15)
    #将numpy array的图片格式转为PIL的图片格式
    img_pil = Image.fromarray(img)
    #创建画板
    draw_ = ImageDraw.Draw(img_pil)
    #在图片上绘制中文
    draw_.text((x+2,y),chinese,font=font,fill=(b,g,r,a))
    #将图片转为numpy array的数据格式
    img = np.array(img_pil)
    cv2.imwrite("{}_{}.jpg".format(i,j),img)
    return img