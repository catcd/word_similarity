**Bài tập lập trình môn học Xử lý ngôn ngữ tự nhiên và ứng dụng**
================
Đo lường độ tương tự ngữ nghĩa và nhận diện quan hệ ngữ nghĩa của từ sử dụng Word Embeddings

Yêu cầu cài đặt 
---------------
Chương trình được cài đặt sử dụng Python 3. Tham khảo: https://www.python.org/downloads/

Các thư viện:

1. **Tensorflow** GPU 1.4.0:

> pip install tensorflow-gpu==1.4.0

If GPU is not available, tensorflow CPU can be used instead:
> pip install tensorflow==1.4.0

2. **Numpy**:
> pip install numpy

3. **Sklearn**:
> pip install scikit-learn

Mô tả cấu trúc của chương trình
----------------
```
word_similarity/            # thử mục gốc của chương trình
|-- data/
|   |-- trained_weight/     # thư mục lưu tham số của mô hình đã huấn luyện
|   |-- ViCon-400/          # chứa dữ liệu được cho
|   |-- ViSim-400/          # chứa dữ liệu được cho
|   `-- w2v/                # chứa bộ word embeddings được cho
|-- k_nearest_words.py      # cài đặt của thuật toán tìm k từ gần nhất với từ w (Bài 2)
|-- mlp_run.py              # chương trình chạy thuật toán huấn luyện mô hình Multi-layer Perceptron (Bài 3)
|-- my_lib
|   |-- dataset.py          # chứa mô-đun xử lý dữ liệu
|   |-- deep_model.py       # chứa mô hình Multi-layer Perceptron (Bài 3)
|   |-- metrics.py          # chứa cài đặt của các độ đo khoảng cách (Bài 1)
|   `-- utils.py            # chứa các hàm tiện ích
`-- README.md               # tài liệu chương trình
```

Chi tiết cài đặt các yêu cầu
----------------
### 1. Yêu cầu 1
Cài đặt chương trình đo độ tương tự ngữ nghĩa của cặp từ sử dụng pre-trained word embeddings được cho sẵn ở ```data/w2v/word2vec.vec```.

Tệp ```my_lib/metrics.py``` chứa cài đặt của các độ đo khoảng cách sau:
- Cosine
- Dice
- Euclidean


### 2. Yêu cầu 2
Tìm $k$ từ gần nhất với từ w.


Tệp ```k_nearest_words.py``` chứa cài đặt của thuật toán này sử dụng cấu trúc dữ liệu Heap.
Để chạy thuật toán với tham số k (ví dụ 10) và w (ví dụ "đồng_cỏ"), gõ lệnh sau tại thư mục gốc:

    python k_nearest_words.py -k 10 -w "đồng_cỏ"

Lưu ý: Nếu w không có trong từ điển thì chương trình sẽ báo ```"WORD NOT IN VOCAB"```.
   

### 3. Yêu cầu 3
Cài đặt mô hình mạng Multi-layer Perceptron để nhận diện cặp từ có quan hệ đồng nghĩa và trái nghĩa sử dụng bộ dữ liệu ViCon-400.

#### 3.1. Mô tả mô hình Multi-layer Perceptron
Đầu vào của mô hình là word_embeddings của hai từ và nhãn là 0 (đồng nghĩa) hoặc 1 (trái nghĩa). Mô hình gồm có N tầng ẩn, mỗi tầng có M units (N và M là các siêu tham số tùy biến). Tại mỗi tầng, hàm kích hoạt được sử dụng là ReLU. Để hạn chế hiện tượng overfitting, phương pháp Dropout cũng được áp dụng sau mỗi tầng ẩn. Ở tầng đầu ra, các giá trị được đưa qua tầng Softmax để thu được một phân bố. Mô hình sử dụng hàm mất mát là Cross Entropy và thuật toán tối ưu là Adam.

Để thực nghiệm thêm, ở tầng đầu vào, ngoài word embeddings của hai từ (đặt là we1 và we2), một số đặc trưng khác có thể tùy chọn để sử dụng thêm như:
- Tổng: we1 + we2
- Trị tuyệt đối của hiệu: |we1 - we2|
- Nhân từng thành phần: we1 * we2

#### 3.2. Thực nghiệm
##### 3.2.1. Dữ liệu
Để huấn luyện và đánh giá mô hình, bộ dữ liệu ViCon-400 được chia thành 2 phần một cách ngẫu nhiên: 90% dùng để huấn luyện và 10% dùng để đánh giá.

##### 3.2.2. Huấn luyện mô hình
Để chạy thuật toán huấn luyện mô hình với các siêu tham số mặc định, gõ lệnh sau tại thư mục gốc:
  
  python mlp_run.py

Danh sách các siêu tham số có thể tùy biến:

```
python mlp_run.py -h
usage: mlp_run.py [-h] [-i I] [-bs BS] [-e E] [-id ID] [-sum SUM] [-sub SUB]
                  [-mul MUL] [-hd HD]

MLP for Synonym Antonym prediction

optional arguments:
  -h, --help  show this help message and exit
  -i I        Job identity
  -bs BS      Batch size
  -e E        Number of epochs
  -id ID      Input w2v dim
  -sum SUM    Use sum vector
  -sub SUB    Use subtract vector
  -mul MUL    Use multiplication vector
  -hd HD      Hidden layer configurations
```

Kết quả độ chính xác trên tập kiểm tra sẽ được báo cáo ngay sau quá trình huấn luyện mô hình hoàn tất.
