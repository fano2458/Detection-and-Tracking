import cv2


def draw_line(image, xf1, yf1, xf2, yf2):
    w = image.shape[1]
    h = image.shape[0]
    
    start_point = (int(w*xf1), int(h*yf1) )
    end_point = (int(w*xf2), int(h*yf2) )
    
    cv2.line(image, start_point, end_point, (255,0,0), 4)


def writes_area_text(image, text, xf1, yf1):
    w = image.shape[1]
    h = image.shape[0]
    
    start_point = (int(w*xf1), int(h*yf1) )
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (255,100,100)
    thickness = 2
    
    cv2.putText(image, text, 
                start_point, font, fontScale, (0,0,0), thickness+3)
    
    cv2.putText(image, text, 
                start_point, font, fontScale, color, thickness)



def which_area(image, midx, midy):
    w = image.shape[1]
    h = image.shape[0]
    xf = midx/w
    yf = midy/h
    
    x1, x2, x3, x4, x5, x6 = 0.10, 0.30, 0.35, 0.55, 0.65, 0.85
    
    y1 = 0.0*xf + 0.2 # Top-left line
    y2 = -0.444*xf + 0.294 # Top-middle line
    y3 = 2.75*xf + -0.025 # Left line
    y4 = -1.0*xf + 1.1 # Bottom line
    y5 = 1.0*xf + -0.2 # Middle Line
    
    if xf <= x1:
        if yf <= y1: # Top-left line
            area = "A2"
        else:
            area = "Register"
    elif xf > x1 and xf <= x2:
        if yf <= y2: # Top-middle line
            area = "A2"
        elif yf <= y3: # Left line
            area = "A3"
        else:
            area = "Register"
    elif xf > x2 and xf <= x3:
        if yf <= y2: # Top-middle line
            area = "A2"
        elif yf <= y4: # Bottom line
            area = "A3"
        else:
            area = "Entrance"
    elif xf > x3 and xf <= x4:
        if yf <= y2: # Top-middle line
            area = "A2"
        elif yf <= y5: # Middle Line
            area = "A1"
        elif yf <= y4: # Bottom line
            area = "A3"
        else:
            area = "Entrance"
    elif xf > x4 and xf <= x5:
        if yf <= y5: # Middle Line
            area = "A1"
        elif yf <= y4: # Bottom line
            area = "A3"
        else:
            area = "Entrance"
    elif xf > x5 and xf <= x6:
        if yf <= y4: # Bottom line
            area = "A1"
        else:
            area = "Entrance"
    else:
        area = "Entrance"
    
    return area


