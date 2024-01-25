import cv2
import numpy as np
from PIL import ImageFont,ImageDraw,Image

def draw_lp(img:np.ndarray,x1:int,y1:int,x2,y2,score,lp_chinese_line,cid,frameid,process_id,res_num): 
    
    '''画框'''
    color = (0, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
    
    '''写score'''
    cv2.putText(img, str(round(score, 2)),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)
    '''写汉字'''
    #导入字体文件
    fontpath = "./fonts/simsun.ttc"
    #设置字体的颜色
    b,g,r,a = 0,255,0,0
    #设置字体大小
    font = ImageFont.truetype(fontpath,40)
    #将numpy array的图片格式转为PIL的图片格式
    img_pil = Image.fromarray(img)
    #创建画板
    draw_ = ImageDraw.Draw(img_pil)
    #在图片上绘制中文
    draw_.text((x1+5,y1),lp_chinese_line,font=font,fill=(b,g,r,a))
    #将图片转为numpy array的数据格式
    img = np.array(img_pil)
    cv2.imwrite("c{}_f{}_res_num{}_P{}_.jpg".format(cid,frameid,res_num,process_id),img)
