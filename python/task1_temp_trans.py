
def temp_trans():
    reset = True
    while reset:
        temp = input('请输入温度值：')
        if temp[-1] in ['F', 'C']:
            if temp[-1] == 'F':
                C = 5/9 * (float(temp[:-1]) - 32.0)
                print(f"转换结果：{C:.2f}°C")
            elif temp[-1] == 'C':
                F = 9/5 * float(temp[:-1]) + 32.0
                print(f"转换结果：{F:.1f}°F")
        else:
            print("输入格式错误！")
        
        while True:
            key = input("继续？(y/n):")
            if key == 'n' or key == 'N':
                reset = False
                break
            elif key == 'y' or key == 'Y':
                break
            else:
                print("输入错误！")

if __name__ == '__main__':

    temp_trans() 

