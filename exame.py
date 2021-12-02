import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from io import BytesIO

import requests

url = 'https://unsplash.com/photos/boaDpmC-_Xo/download?ixid=MnwxMjA3fDB8MXxhbGx8MXx8fHx8fDJ8fDE2MzQ2ODI0MzQ&force=true&w=640'
response = requests.get(url)

image = Image.open(BytesIO(response.content)).convert()
image

img_arr = np.copy(image)
# img_arr.shape
img_gray_average = np.mean(img_arr, axis=2)

# img_gray_average

plt.imshow(img_gray_average, cmap='gray')
plt.axis('off')
plt.show()

new_img_gray_average = np.empty(shape=img_arr.shape, dtype=np.uint8)
new_img_gray_average[:,:,0] = img_gray_average
new_img_gray_average[:,:,1] = img_gray_average
new_img_gray_average[:,:,2] = img_gray_average

new_img_gray_average

img_res = Image.fromarray(new_img_gray_average)
img_res

img_gray_average.shape

pesos = [0.2126, 0.7152, 0.0722]

res = np.array(img_arr * pesos, dtype=np.uint8)
y = np.array(np.sum(res, axis=2), dtype=np.uint8)

concat = np.array(np.concatenate((img_gray_average,y), axis=1), dtype=np.uint8)

img_ones = Image.fromarray(concat)
img_ones

print(res[0,0,:])
print(img_arr[0,0,:])

res.shape

np.sum(res, axis=2).shape

full = 1408 * 2400 * 3
original = 2750 *  4687 * 3

print(f"full: {full}, original: {original} {original/full}")

new_array_test = np.sum(res, axis=2)
type(new_array_test)

plt.imshow(new_array_test, cmap='gray')
plt.show()

img_gamma = np.copy(image)
img_gamma

img_gamma_final = img_gamma
img_gamma_final = Image.fromarray(img_gamma)
plt.imshow(img_gamma)
plt.show()

maior_gamma = np.where(img_gamma > .04045)
img_gamma_final[maior_gamma] = ((img_gamma[maior_gamma] + 0.055) / 1.055) ** 2.4
menor_gamma = np.where(img_gamma <= .04045)
img_gamma_final[menor_gamma] = img_gamma[menor_gamma] / 12.92
img_gamma_final = (img_gamma_final)/4 
plt.imshow(img_gamma_final)
plt.axis('off')
plt.show()

vermelho = img_gamma_final[:,:,0]
verde = img_gamma_final[:,:,1]
azul = img_gamma_final[:,:,2]

aproximacao  = ((vermelho *0.299) + (verde * 0.587) + (azul * 0.114 ))
plt.imshow(aproximacao)
plt.axis('off')
plt.show()