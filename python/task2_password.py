import string
import random

def password():
    reset = True
    while reset:
        
        while True:
            n = input('密码长度：')
            if n.isnumeric():
                n = int(n)
                if n > 0:
                    break
            else:
                print("输入错误！")

        while True:
            is_upper = input('包含大写字母？(y/n):')
            if is_upper == 'y' or is_upper == 'n':
                break
            else:
                print("输入错误！")

        while True:
            is_sepcial = input('包含特殊字符？(y/n):')
            if is_sepcial == 'y' or is_sepcial == 'n':
                break
            else:
                print("输入错误！")

        # password 
        chara = string.ascii_lowercase + string.digits
        if is_upper == 'y' or is_upper == 'Y':
            chara += string.ascii_uppercase
        if is_sepcial == 'y' or is_sepcial == 'Y':
            chara = string.printable
        
        password = random.choices(chara, k=int(n))
        password = ''.join(password)
        
        print(f"生成密码：{password}")

        while True:
            key = input("再次生成？(y/n):")
            if key == 'n':
                reset = False
                break
            elif key == 'y':
                break
            else:
                print("输入错误！")

if __name__ == '__main__':
    password()