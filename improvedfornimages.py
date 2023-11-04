                        #  ============>   Required Packages     <===============   #

import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import numpy as np
import random
import math

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import base64
import os

                        #  ============>   AES Encryption    <==============   #


def encrypt_aes_cbc(shivkey, plaintext):
    shivkey = shivkey.ljust(32, b'\0')  # Ensure the shivkey is 32 bytes (256 bits)
    iv = os.urandom(16)  # Generate a random 16-byte IV
    cipher = Cipher(algorithms.AES(shivkey), modes.CBC(iv),
                    backend=default_backend())
    encryptor = cipher.encryptor()

    # Pad the plaintext to be a multiple of 16 bytes
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext) + padder.finalize()

    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    encrypted_str = base64.b64encode(iv + ciphertext).decode('utf-8')
    return encrypted_str


                        #  ============>   DES Encryption    <==============   #


def decrypt_aes_cbc(shivkey, ciphertext):
    shivkey = shivkey.ljust(32, b'\0')  # Ensure the shivkey is 32 bytes (256 bits)

    # Decode the base64-encoded ciphertext
    ciphertext = base64.b64decode(ciphertext.encode('utf-8'))
    iv = ciphertext[:16]  # Extract the IV from the ciphertext
    ciphertext = ciphertext[16:]  # Extract the actual ciphertext

    cipher = Cipher(algorithms.AES(shivkey), modes.CBC(iv),
                    backend=default_backend())
    decryptor = cipher.decryptor()

    decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()

    # Unpad the decrypted data
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()

    decrypted_str = unpadded_data.decode('utf-8')
    return decrypted_str



                        #  ============>   PARTIONING THE DATA FOR IMAGES   <==============   #



def partition_sizes(n, min_value, max_value, min_size_of_partition, text_len):
    partition_points = [random.randint(min_value, max_value)]
    while len(partition_points) < n - 1:
        new_point = random.randint(min_value, max_value)
        if all(abs(new_point - point) > min_size_of_partition for point in partition_points):
            partition_points.append(new_point)
    partition_points.sort()
    partition_points.insert(0, 0)
    partition_points.append(100)

    partitions = []
    for i in range(n):
        start = partition_points[i]
        end = partition_points[i + 1]
        partition = end - start
        partitions.append(partition)

    total_partition_length = sum(partitions)
    scaling_factor = text_len / total_partition_length

    partitions = [math.ceil(partition * scaling_factor) for partition in partitions]

    # Adjust the last partition to make sure the sum matches text_len
    partitions[-1] += text_len - sum(partitions)

    print("Sum:", sum(partitions))
    print("Partitions:", partitions)

    return partitions


                            #  ============>  UTILITY FUNCTIONS  <==============   #



def logistic_map(x, r):
    return r * x * (1 - x)
def generate_logistic_map_values(x0, r, num_values):
    values = []
    x = x0
    for _ in range(num_values):
        x = logistic_map(x, r)
        values.append(x)
    return values
def generate_random_parameters():
    x0 = random.uniform(0, 1)
    r = random.uniform(3.57, 4.0)
    return x0, r
def generate_random_values_using_logistic_map(n):
    x0, r = generate_random_parameters()
    random_values = generate_logistic_map_values(x0, r, n)
    return random_values

def sort_and_get_indices(values):
    sorted_values = sorted(values)
    indices = [sorted_values.index(value) for value in values]
    return indices
def get_position(n,y):
    random_values = generate_random_values_using_logistic_map(n)
    num_list = sort_and_get_indices(random_values)
    position_list = []
    for i in range(n):
        var = num_list[i]
        i = math.floor(var/(y+1))
        j = var % (y+1)
        position_list.append((i, j))
    return position_list

def calculate_psnr(original_image, encrypted_image):
    mse = np.mean((original_image - encrypted_image) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def text_to_binary(text):
    binary_string = ''.join(format(ord(char), '08b') for char in text)
    return binary_string

def normalize_length(text1, text2):
    max_length = max(len(text1), len(text2))
    text1 = text1.ljust(max_length)
    text2 = text2.ljust(max_length)
    return text1, text2

def calculate_ber_from_text(input_text1, input_text2):
    binary_string1 = text_to_binary(input_text1)
    binary_string2 = text_to_binary(input_text2)

    binary_string1, binary_string2 = normalize_length(
        binary_string1, binary_string2)

    differing_bits = sum(bit1 != bit2 for bit1, bit2 in zip(
        binary_string1, binary_string2))

    ber = differing_bits / len(binary_string1)
    return ber


d = {}
c = {}

for i in range(255):
    d[chr(i)] = i
    c[i] = chr(i)


                           #  ============>   START    <==============   #


text = input(f"Enter text to hide:")
shivkey = b'Thirty-two byte key for AES-256'

encrypted_text=encrypt_aes_cbc(shivkey,text.encode('utf-8'))
print("Encrypted Text : ",encrypted_text)

temp=len(encrypted_text)
numberofimages=len(str(temp))
charater_count=partition_sizes(numberofimages,1,100,1,temp)
random_values = generate_random_values_using_logistic_map(numberofimages)
image_order = sort_and_get_indices(random_values)

charater_image_map=[]
for i in range(numberofimages):
    charater_image_map.append((charater_count[i],image_order[i]))
print("No. of Characters to Image Mapping : ",charater_image_map)
start=0
counttext=0


                            #  ============>  ENCODING DATA TO IMAGES    <==============   #


decls=[]
#Doing Encryption in n images as per mapping.
while(start<numberofimages):
    x = cv2.imread(f"{charater_image_map[start][1]}.png")
    j = x.shape[1]
    ls=get_position(charater_image_map[start][0],j-1)
    decls.append(ls)
    z = 0
    temptext=encrypted_text[counttext:counttext+charater_image_map[start][0]]
    for i in range(charater_image_map[start][0]):
        x[ls[i][0], ls[i][1], z] = d[temptext[i]] ^ x[ls[i][0],ls[i][1],z]
        z = (z + 1) % 3
    cv2.imwrite(f"{charater_image_map[start][1]}encrypted_img.png", x)
    counttext+=charater_image_map[start][0]
    start+=1
print("Data Hiding in Image completed successfully.")

PSNR=[]
MSE=[]
SSIM=[]

ch = input("\nEnter any value to decrypt : ")


                            #  ============>  DECODING DATA FROM IMAGES    <==============   #


start=0
counttext=0
decrypt = ""
ls=decls
print("\nList used in decryption is:",ls)
while(start<numberofimages):
    temptext=""
    x = cv2.imread(f"{charater_image_map[start][1]}encrypted_img.png")
    original_image=cv2.imread(f"{charater_image_map[start][1]}.png")
    z = 0 
    for i in range(charater_image_map[start][0]):
        temptext += c[x[ls[start][i][0], ls[start][i][1], z] ^ original_image[ls[start][i][0],ls[start][i][1],z]]
        z = (z + 1) % 3
    decrypt+=temptext
    mse_value = mean_squared_error(x,original_image)
    ssim_value = ssim(original_image,x,win_size=3)
    x = x.astype(original_image.dtype)
    psnr_value = calculate_psnr(original_image,x)
    PSNR.append(psnr_value)
    MSE.append(mse_value)
    SSIM.append(ssim_value)
    start+=1


                    #  ============>   FINAL RESULTS AND TESTS    <==============   #


decrypted_text=decrypt_aes_cbc(shivkey,decrypt)
print("\nDecoded Text:\n",decrypt,"\n")

print("Initial text was   : ",text)
print("Decrypted text was : ",decrypted_text)
ber = calculate_ber_from_text(text,decrypted_text)
print(f"Bit Error Rate (BER): {ber:.4f}")

for i in range(len(PSNR)):
    print("Values for image: ",i+1)
    print(f"PSNR value is {PSNR[i]} dB")
    print(f'MSE: {mse_value:.2f}')
    print(f'SSIM: {ssim_value:.8f}')
