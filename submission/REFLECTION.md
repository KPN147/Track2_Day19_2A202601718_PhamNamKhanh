# Reflection — Lab 19

**Tên:** Phạm Nam Khánh
**Cohort:** A20-K3
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên 50 golden queries, hybrid có Precision@10 trung bình cao nhất: **78,6%**,
so với BM25 **77,8%** và vector **73,2%**. Với `exact`, BM25 và hybrid cùng đạt
**96,7%** vì thuật ngữ kỹ thuật xuất hiện nguyên văn. Với `mixed`, hybrid thắng
ở **100%** nhờ RRF kết hợp keyword với semantic signal. Riêng `paraphrase`,
kết quả thực nghiệm không giống kỳ vọng:
BM25 đạt **33,3%**, hybrid **32,0%**, còn vector chỉ **24,0%**. Nguyên nhân là
lab lite dùng `BAAI/bge-small-en-v1.5`, model thiên về tiếng Anh nên hiểu
paraphrase tiếng Việt chưa tốt; một model multilingual như `bge-m3` có thể đảo
kết quả này.

Không dùng hybrid khi query là mã lỗi, ID hoặc thuật ngữ cần exact match:
BM25 nhanh và dễ giải thích hơn. Pure vector phù hợp cho truy vấn theo nghĩa
khi corpus đa ngôn ngữ và embedding model đã được đánh giá tốt. Hybrid cũng
không đáng dùng nếu quality tăng không bù được latency; benchmark hiện tại cho
hybrid P99 **79,9 ms**, cao hơn ngưỡng 50 ms.

---

## Điều ngạc nhiên nhất khi làm lab này

Điều ngạc nhiên nhất là vector search không tự động thắng paraphrase: chất lượng
embedding phụ thuộc mạnh vào ngôn ngữ của model. Hybrid ổn định hơn về quality,
nhưng cái giá latency thể hiện rất rõ ở tail P99.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work: Không
