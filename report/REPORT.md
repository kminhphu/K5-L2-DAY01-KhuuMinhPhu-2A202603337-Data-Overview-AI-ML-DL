# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 2026-11-06**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics: 3.13.16/2.11.0+cu130/8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> \\\*\\\*Cách dùng mẫu này:\\\*\\\* copy/điền trên máy, lưu thành `REPORT.md`, rồi \\\*\\\*upload\\\*\\\* vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1\\\_lab\\\_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp \\\*\\\*link repo\\\*\\\* trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

|Từ|Nghĩa ngắn|
|-|-|
|Prediction|Kết quả model vừa đoán ra; chưa phải nhãn đúng|
|Ground truth|Nhãn do người gán theo quy tắc|
|Taxonomy|Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80)|
|Checkpoint|File model `.pt` đã huấn luyện sẵn|
|Guideline|Quy tắc gán nhãn do người viết|
|Annotator|Người gán nhãn|
|Reviewer / QC|Người kiểm tra nhãn|
|Escalation|Ca khó → hỏi Lab Coach, không tự bịa|
|Model score|Số “tin” của model; không phải điểm chất lượng nhãn người|

## 1\. Phân loại ảnh

**Mở file:** `day1\\\_lab\\\_outputs/classification\\\_predictions.json` và `visuals/classification\\\_top5.png`.  
Chỉ dùng ảnh `sample\\\_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class\\\_id`= 468 ; `class\\\_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy\\\_name`= ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Tra lời: Checkpoint yolo11n-cls.pt xếp hạng các lớp ImageNet-1K cho toàn bộ ảnh (không phải từng object).
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Trả lời: Class list đi kèm taxonomy/checkpoint (ImageNet-1K với yolo11n-cls.pt), không phải model tự nghĩ ra tên lớp.
4. Vì sao cần ghi cả `class\\\_id`, `class\\\_name` và `taxonomy\\\_name`, không chỉ ghi mỗi tên lớp?
Trả lời: Cần cả ba vì: class\_id ổn định cho máy; class\_name dễ đọc cho người; taxonomy\_name cho biết đang dùng bộ lớp nào (tránh nhầm khi đổi dataset/checkpoint).
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Trả lời: Guideline cần nói rõ chọn lớp theo tiêu chí nào khi ảnh nhiều chủ đề (ví dụ: chủ thể lớn nhất / chủ đề chính của cảnh / lớp xuất hiện rõ nhất), để mọi annotator chọn thống nhất một lớp.
6. Vì sao không được lấy `score` của model làm ground truth?
Trả lời: Vì score chỉ là độ tin của model trên prediction; nó không được người xác nhận theo guideline, và top-1 ImageNet trên ảnh COCO có thể trông “lạ” nhưng vẫn chỉ là prediction.

## 2\. Phát hiện vật thể

**Mở file:** `day1\\\_lab\\\_outputs/detection\\\_predictions.json` và `visuals/detection\\\_predictions.png`.  
Chỉ dùng ảnh `sample\\\_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class\\\_name`=bus ; `score`= 0.912558 ; `bbox\\\_xyxy`=\[93.17, 187.95, 223.01, 320.91] ; `bbox\\\_width`= 129.84 ; `bbox\\\_height`= 132.96
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Trả lời: Lớp bus, score 0.912558; box từ góc trên-trái khoảng pixel (93.17, 187.95) đến góc dưới-phải (223.01, 320.91); rộng khoảng 130 px, cao khoảng 133 px.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Trả lời: threshold=0.20: 17 vật thể - threshold=0.60: 6 vật thể
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Trả lời: Reviewer phải xem thêm nhiều hộp hơn, kể cả hộp nhiễu/score thấp, nên khối lượng kiểm tra tăng.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Trả lời: Quy tắc box: Box phải ôm sát vật thể, không cắt mất phần nhìn thấy của object; không vẽ một box chung cho hai object tách rời.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trả lời: Trên ảnh kitchen, vật bị cắt mép bàn/đồ dùng gần biên ảnh: tôi chọn guideline nếu rule đã có (ví dụ “còn ≥ 50% object trong khung thì vẫn gán”); nếu không chắc object là gì vì che khuất nặng thì escalation hỏi Lab Coach, không tự bịa lớp.

## 3\. Phân đoạn theo từng đối tượng

**Mở file:** `day1\\\_lab\\\_outputs/segmentation\\\_predictions.json` và `visuals/segmentation\\\_prediction.png`.  
Chỉ dùng ảnh `sample\\\_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance\\\_id`= traffic-001; `class\\\_name`= bus; `score`= 0.925745; số điểm polygon=`polygon\\\_point\\\_count`= 120 ; 3–5 điểm đầu của `polygon\\\_xy`=\[148.0,189.0],\[147.0,190.0],\[145.0,190.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Trả lời: Polygon bám theo biên thật của object (cong/mép), còn box chỉ là hình chữ nhật bao ngoài nên mất chi tiết hình dạng.
3. `instance\\\_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
Trả lời: instance\_id dùng để phân biệt từng object trong output lab (vd. hai bowl khác nhau là traffic-001 và traffic-002). Nó không phải mã lớp (class\_id) và không phải ID theo dõi theo thời gian (tracking).
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Trả lời: Quy tắc biên mask: Mask phải đi sát biên nhìn thấy của instance; không tràn sang object bên cạnh; vùng bị che khuất thì chỉ vẽ phần còn nhìn thấy (trừ khi guideline nói khác).
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trả lời: Chỗ hai đồ dùng chạm nhau trên bàn kitchen: nếu guideline chưa nói rõ tách biên thế nào thì escalation; nếu guideline đã bảo “tách theo khe nhìn thấy giữa hai object” thì làm theo guideline.

## 4\. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

|Tác vụ|Ground truth trông như thế nào? (đơn vị nhãn)|Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON|Annotator làm gì?|Reviewer xem gì?|
|-|-|-|-|-|
|Phân loại ảnh|Một lớp cho cả ảnh theo taxonomy ImageNet-1K|Top-1 có thể không khớp trực giác cảnh giao thông (prediction ≠ GT)|Đọc guideline chọn đúng 1 lớp cho ảnh|Đúng taxonomy, đúng phạm vi cấp ảnh, không chép score thành nhãn|
|Phát hiện vật thể|Mỗi object: class + box xyxy|Hạ threshold ra thêm hộp nhiễu; object cắt mép|Vẽ box theo rule, đủ object trong guideline|Box có ôm sát không, thiếu/thừa object, ca che khuất|
|Instance segmentation|Mỗi instance: class + polygon|Biên hai vật chạm nhau / vùng mờ|Vẽ mask bám biên từng instance|Mask có tràn object khác không, đủ instance cùng lớp|

## 5\. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Trả lời: Chỉ dùng ảnh COCO công khai có sẵn trong notebook; không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Trả lời: Dừng thao tác và báo Lab Coach / GV phụ trách lab.
## 6\. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

* \[x] `REPORT.md` (file này đã điền)
* \[ ] `day1\\\_lab\\\_outputs/classification\\\_predictions.json`
* \[ ] `day1\\\_lab\\\_outputs/detection\\\_predictions.json`
* \[ ] `day1\\\_lab\\\_outputs/segmentation\\\_predictions.json`
* \[ ] `day1\\\_lab\\\_outputs/IMAGE\\\_ATTRIBUTION.md`
* \[ ] `day1\\\_lab\\\_outputs/visuals/classification\\\_top5.png`
* \[ ] `day1\\\_lab\\\_outputs/visuals/detection\\\_predictions.png`
* \[ ] `day1\\\_lab\\\_outputs/visuals/segmentation\\\_prediction.png`
* \[ ] Ô validation cuối notebook báo `PASS`
* \[x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
* \[ ] Đã nộp **link repository** trên LMS

