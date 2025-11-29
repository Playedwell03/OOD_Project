import os

input_dir = 'final_data_origin_for_learn/train'

img_dir = os.path.join(input_dir, 'images')
label_dir = os.path.join(input_dir, 'labels')

imgcount = 0
lbcount = 0

files = os.listdir(input_dir)
for j in files:
    imgcount += 1
print(input_dir, ' 폴더의 파일 개수는', imgcount, '개입니다.')

# files = os.listdir(img_dir)
# for j in files:
#     imgcount += 1
# print(img_dir, ' 폴더의 파일 개수는', imgcount, '개입니다.')

# files = os.listdir(img_dir)
# for j in files:
#     lbcount += 1
# print(img_dir, ' 폴더의 파일 개수는', lbcount, '개입니다.')