import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct,idct
import heapq
from collections import defaultdict

#Compression Ratio Function
def compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

#Peak Signal-to-Noise Ratio Function
def psnr(original_img, compressed_img):
    mse = np.mean((original_img - compressed_img) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

q_factors = [1, 5, 32, 75]
for q_fac in q_factors:

    #Encoding Steps(ignore header and Huffman Table)----------------------------------------------------------------------
    #First Step, Read an image in grayscale
    img = cv.imread('/Users/a/Documents/middleExam/lena.png',cv.IMREAD_GRAYSCALE)
    original_size = img.nbytes
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.show()
    print("1th Finish reading an image in grayscale")
    print(f"  > Original Image Size: {original_size} bytes\n\n")
    print("1th Finish reading an image in grayscale")


    #Second Step, Process 3-8 for each 8x8 sub-images (blocks)
    # 블록 크기 설정
    block_size = 8
    BLOCK_SIZE = 8
    # 이미지 크기 가져오기
    h, w = img.shape
    # 블록 단위로 이미지를 분할
    blocks = [img[i:i+block_size, j:j+block_size] for i in range(0, h, block_size) for j in range(0, w, block_size)]
    # 결과 확인
    print(f"Total number of blocks: {len(blocks)}")
    # 첫 번째 블록 출력 (예시)
    plt.imshow(blocks[0], cmap='gray')
    plt.title('First 8x8 Block')
    plt.show()
    print(len(blocks[0]))
    print(blocks[0])
    print("2th Start processing 3-8 for each 8x8 sub-images (blocks)\n\n")


    #Third Step, Shift the intensity level
    #Change the range of pixel values from [0, 255] to [-128, 127]
    shifted_blocks = [block.astype(np.float32) - 128 for block in blocks]
    # 첫 번째 shifted block을 표시
    plt.imshow(shifted_blocks[0], cmap='gray', vmin=-128, vmax=127)
    plt.title('First 8x8 Block after Intensity Shift')
    plt.colorbar()
    plt.show()
    print(len(shifted_blocks[0]))
    print(shifted_blocks[0])
    print("3th Finish shifting intensity level")


    #Fourth Step, Forward Discrete Cosine Transform (DCT)
    #apply DCT and devide image into 8x8 blocks
    def apply_dct(block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    dct_blocks = [apply_dct(block) for block in shifted_blocks]
    plt.imshow(dct_blocks[0], cmap='gray')
    plt.title('First 8x8 Block after DCT ')
    plt.colorbar()
    plt.show()
    print(dct_blocks[0])
    print("4th Finish DCT")


    #Fifth Step, Quantize
    base_quantization_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    # DCT 블록 양자화
    quantization_table = base_quantization_table * (100 / q_fac)
    quantized_blocks = [np.round(block / quantization_table) for block in dct_blocks]
    # 첫 번째 양자화 블록 표시
    plt.imshow(quantized_blocks[0], cmap='gray')
    plt.title('First 8x8 Block after Quantization')
    plt.colorbar()
    plt.show()
    print(quantized_blocks[0])
    print(quantized_blocks[1])
    print("5th Finish Quantization\n\n")


    #Sixth Step, Scan in zigzag order
    zigzag_indices = np.array([
        0, 1, 8, 16, 9, 2, 3, 10, 
        17, 24, 32, 25, 18, 11, 4, 5, 
        12, 19, 26, 33, 40, 48, 41, 34, 
        27, 20, 13, 6, 7, 14, 21, 28, 
        35, 42, 49, 56, 57, 50, 43, 36, 
        29, 22, 15, 23, 30, 37, 44, 51, 
        58, 59, 52, 45, 38, 31, 39, 46, 
        53, 60, 61, 54, 47, 55, 62, 63
    ])
    def zigzag_order(block):
        flattened = block.ravel()
        zigzagged = np.array([flattened[index] for index in zigzag_indices])
        return zigzagged
    zigzag_blocks = [zigzag_order(block) for block in quantized_blocks]
    print("  > Result of zigzag scan(first block) : ", "\n", zigzag_blocks[0], "\n\n")
    print("  > Result of zigzag scan(first block) : ", "\n", zigzag_blocks[1], "\n\n")
    print(len(zigzag_blocks[0]))
    print("6th Finish Zigzag\n\n")


    #Seventh Step, Run-length encoding
    def run_length_encode(data):
        rle = []
        last_val = None
        count = 0
        for val in data:
            if val == last_val:
                count += 1
            else:
                if last_val is not None:
                    rle.append((count, last_val))    
                last_val = val
            count = 1
            rle.append((count, last_val))
        return rle
    rle_blocks = [run_length_encode(zigzag_block) for zigzag_block in zigzag_blocks]
    print("  > rle_blocks[0] : ", rle_blocks[0])
    print("  > rle_blocks[0] : ", rle_blocks[1])
    print("7th Finish Run-Length Encoding\n\n")


    #Eighth Step, Huffman coding
    def build_huffman_tree(frequency):
        heap = [[wt, [sym, ""]] for sym, wt in frequency.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            low = heapq.heappop(heap)
            high = heapq.heappop(heap)
            for pair in low[1:]:
                pair[1] = '0' + pair[1]
            for pair in high[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])
        huffman_dict = dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))
        return huffman_dict
    def huffman_encode(data, huffman_dict):
        encoded_data = []
        for count, val in data:
            code = huffman_dict.get((count, val), '')
            encoded_data.append((code, (count, val)))
        return encoded_data
    # Calculate frequency for run-length encoded data
    frequency = defaultdict(int)
    for rle in rle_blocks:
        for count, val in rle:
            frequency[(count, val)] += 1
    # Build Huffman tree and generate Huffman codes
    huffman_dict = build_huffman_tree(frequency)
    # Huffman encode the run-length encoded data
    huffman_encoded_blocks = [huffman_encode(rle, huffman_dict) for rle in rle_blocks]
    print("huffman_encoded_blocks[0] : ", huffman_encoded_blocks[0])
    print("huffman_encoded_blocks[0] : ", huffman_encoded_blocks[1])
    print("8th Finish Huffman Encoding\n\n\n")



    #Decoding Steps(Inverse steps of Encoding)----------------------------------------------------------------------
    #First Step, Huffman decoding
    def huffman_decode(encoded_data, huffman_dict):
        reverse_huffman_dict = {v: k for k, v in huffman_dict.items()}  # 코드를 원래 값으로 매핑
        decoded_data = []
        for code, _ in encoded_data:
            original_value = reverse_huffman_dict[code]
            decoded_data.append(original_value)
        return decoded_data
    # 전체 huffman_encoded_blocks에 대해 허프만 디코딩 수행
    decoded_blocks = [huffman_decode(encoded_block, huffman_dict) for encoded_block in huffman_encoded_blocks]
    # 디코딩된 첫 번째 블록 출력
    print("decoded_blocks[0] : ", decoded_blocks[0])
    print("decoded_blocks[0] : ", decoded_blocks[1])
    print(len(decoded_blocks[0]))
    print("Finish Huffman Decoding\n\n")


    #Second Step, Deconstruct RLE
    def rle_decode(rle_data):
        decoded_data = []  # 디코딩된 결과를 저장할 리스트
        for count, value in rle_data:  # RLE 데이터의 각 쌍에 대해 반복
            # 주어진 횟수만큼 값을 추가
            decoded_data.extend([value] * count)  # `count`만큼 `val`을 확장하여 추가
        #del decoded_data[0]
        return decoded_data  # 디코딩된 결과 반환
    decoded_rle_blocks = [rle_decode(block) for block in decoded_blocks]
    print(decoded_rle_blocks[0])
    print(decoded_rle_blocks[1])
    print(len(decoded_rle_blocks[0]))
    print("Finish Run-Length Decoding\n\n")


    #Third Step, Re-order in zigzag order
    def zigzag_decode(block, block_size):
        # 초기 2D 배열을 생성합니다.
        n = block_size
        result = [[0] * n for _ in range(n)]
        index = 0

        # (x, y)가 2차원 배열의 좌표입니다. (i, j)가 대각선의 인덱스입니다.
        for i in range(2 * n - 1):
            if i % 2 == 0:
                # 대각선이 짝수 번째일 때, 우상향 대각선을 따라 값을 할당합니다.
                x = i if i < n else n - 1
                y = 0 if i < n else i - n + 1
                while x >= 0 and y < n:
                    result[y][x] = block[index]
                    index += 1
                    x -= 1
                    y += 1
            else:
                # 대각선이 홀수 번째일 때, 좌하향 대각선을 따라 값을 할당합니다.
                x = 0 if i < n else i - n + 1
                y = i if i < n else n - 1
                while x < n and y >= 0:
                    result[y][x] = block[index]
                    index += 1
                    x += 1
                    y -= 1
        return result

    # 전체 RLE 디코딩된 블록에 대해서 지그재그 디코딩을 수행합니다.
    block_size = 8  # 8x8 블록 크기로 가정합니다.
    zigzag_decoded_blocks = [zigzag_decode(block, block_size) for block in decoded_rle_blocks]
    print("zigzag_decoded_blocks[0] : \n", zigzag_decoded_blocks[0], "\n")
    print("zigzag_decoded_blocks[1] : \n", zigzag_decoded_blocks[1], "\n")
    print(len(zigzag_decoded_blocks[0]))
    print("Finish Re-order in zigzag order\n\n")


    #Fourth Step, Dequantize
    quantization_table = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99
    ]).reshape(8, 8)
    def dequantize(block, quant_table):
        # 각 블록의 원소에 대응하는 양자화 테이블의 값으로 곱합니다.
        return block * quant_table
    # 각 지그재그 디코딩된 블록을 dequantize합니다.
    dequantized_blocks = [dequantize(np.array(block), quantization_table) for block in zigzag_decoded_blocks]
    plt.imshow(dequantized_blocks[0], cmap='gray')
    plt.title('Dequantize')
    plt.colorbar()
    plt.show()
    print(dequantized_blocks[0])
    print(dequantized_blocks[1])
    print("Finish Dequantization\n\n")


    #Fifth Step, Inverse Discrete Cosine Transform
    def apply_idct(block):
        idct_result = idct(idct(block.T, norm='ortho').T, norm='ortho')
        #print(f"IDCT Result: {idct_result}")  # Debugging: IDCT result
        return idct_result
    # 각 dequantized 블록에 대해 IDCT를 적용합니다.
    idct_blocks = [apply_idct(block) for block in dequantized_blocks]
    plt.imshow(idct_blocks[0], cmap='gray')
    plt.title('Inverse Discrete Cosine Transform')
    plt.colorbar()
    plt.show()
    print("Finish Inverse Discrete Cosine Transform\n\n")


    # Intensity level shifting을 적용하는 함수를 정의합니다.
    def level_shift(block, shift_value=128):
        return block + shift_value
    # 각 IDCT 적용된 블록에 대해 intensity level shifting을 적용합니다.
    shifted_blocks = [level_shift(block) for block in idct_blocks]
    plt.imshow(shifted_blocks[2], cmap='gray')
    plt.title('Intensity level shifting')
    plt.colorbar()
    plt.show()
    print("Finish Shifting intensity level\n\n")


    #Eight Step, Compare with the original image
    image_height = image_width = 64 * 8  # 512
    # 블록들을 이미지로 재조합하는 함수
    def combine_blocks(blocks, blocks_per_row, block_size):
        # 이미지의 높이와 너비
        image_height = blocks_per_row * block_size
        image_width = image_height  # 가정: 이미지는 정사각형이다
        # 빈 이미지를 생성합니다.
        image = np.zeros((image_height, image_width), dtype=np.float32)
        # 각 블록을 올바른 위치에 배치합니다.
        for block_idx, block in enumerate(blocks):
            # 블록의 행과 열 위치를 계산합니다.
            row = (block_idx // blocks_per_row) * block_size
            col = (block_idx % blocks_per_row) * block_size
            # 이미지에 블록을 배치합니다.
            image[row:row+block_size, col:col+block_size] = block
        return image
    # 모든 블록을 하나의 이미지로 합칩니다.
    full_image = combine_blocks(shifted_blocks, blocks_per_row=64, block_size=8)
    # full_image가 최종 이미지입니다. 값을 출력하거나 이미지로 저장할 수 있습니다.
    # 이미지 값을 0과 255 사이로 클리핑합니다.
    full_image = np.clip(full_image, 0, 255).astype(np.uint8)
    # 이미지를 확인합니다(여기서는 값을 출력합니다).
    print(full_image)
    plt.imshow(full_image, cmap='gray')
    plt.title('Final Image')
    plt.colorbar()
    plt.show()



    compressed_img = full_image
    compressed_size = compressed_img.nbytes
    comp_ratio = compression_ratio(original_size, compressed_size)
    psnr_value = psnr(img, full_image)  # 원본 이미지와 비교
    print(f"Compression Ratio at q_factor[{q_fac}] : ", {comp_ratio})
    print(f"PNSR at q_factor[{q_fac}] : ", {psnr_value}, "dB")
    print("Finish Encoding & Decoding at q_factor[", q_fac, "]\n\n\n")

    