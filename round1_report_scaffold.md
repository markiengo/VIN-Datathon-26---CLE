# Khung Báo Cáo Vòng 1

> Ghi chú: đây là file khung nháp để tổng hợp ý và figure plan. Khi review nội dung chính, ưu tiên đọc `round1_report.pdf`, `round1_report.tex`, `eda_round1.ipynb` và `forecasting.ipynb`.

Tài liệu này là khung nội dung để bạn chuyển vào template NeurIPS 4 trang của vòng 1. Nội dung bám trực tiếp rubric trong `Đề thi Vòng 1.pdf` và đồng bộ với hai notebook:

- `eda_round1.ipynb` cho Phần 2
- `forecasting.ipynb` cho Phần 3

## 0. Checklist Trước Khi Nộp

- `submission.csv` có đúng `548` dòng và giữ nguyên thứ tự như `sample_submission.csv`
- Báo cáo giới hạn `<= 4` trang, chưa tính references và appendix
- Report có link GitHub repo và mô tả cách tái lập kết quả
- Report có phần explainability cho forecasting
- Repo public hoặc đã cấp quyền cho ban tổ chức

## 1. Gợi Ý Phân Bổ 4 Trang

| Trang | Nội dung chính | Mục tiêu rubric |
|---|---|---|
| 1 | Business context, data scope, Figure 1-2 từ EDA | Descriptive + Diagnostic |
| 2 | Figure 3-5 từ EDA, action summary | Predictive + Prescriptive |
| 3 | Forecasting pipeline, validation, leakage control | Technical report quality |
| 4 | Explainability, final recommendations, limitations | Explainability + business impact |

## 2. Figure Plan Đề Xuất

| Figure | Nguồn | Thông điệp chính |
|---|---|---|
| Figure 1 | `eda_round1.ipynb` - Demand baseline | Revenue có regime shift sau 2018 và mùa vụ rất rõ |
| Figure 2 | `eda_round1.ipynb` - Traffic vs orders | Revenue đi cùng orders và conversion mạnh hơn raw traffic |
| Figure 3 | `eda_round1.ipynb` - Promotion economics | `fixed promotions` đang phá margin, cần redesign |
| Figure 4 | `eda_round1.ipynb` - Returns & cancellations | `COD` và `wrong_size` là hai điểm rò rỉ doanh thu chính |
| Figure 5 | `eda_round1.ipynb` - Category portfolio | Streetwear là động cơ doanh thu nhưng cũng là cụm rủi ro lớn nhất |
| Figure 6 | `forecasting.ipynb` - CV / SHAP | Pipeline forecasting đúng chiều thời gian và có explainability |

## 3. Abstract Mẫu

Bài làm phân tích hoạt động của một doanh nghiệp thời trang e-commerce tại Việt Nam trên giai đoạn 2012-2022, với trọng tâm là tìm ra các đòn bẩy tăng trưởng và rò rỉ hiệu quả vận hành. Phần EDA chỉ ra bốn phát hiện chính. Thứ nhất, doanh thu bước vào một regime mới từ năm 2019, với mức doanh thu ngày trung bình thấp hơn khoảng `40.6%` so với giai đoạn 2012-2018, nhưng vẫn duy trì mùa vụ mạnh, trong đó tháng 4 cao gấp gần `2.9x` tháng 12. Thứ hai, doanh thu gắn với số đơn hàng mạnh hơn nhiều so với raw traffic (`corr(Revenue, Orders)=0.9377` so với `corr(Revenue, Sessions)=0.3211`), cho thấy tối ưu chuyển đổi quan trọng hơn chỉ tăng traffic. Thứ ba, khuyến mãi kiểu `fixed` tạo doanh thu nhưng đốt margin, với gross profit âm khoảng `230.5M VND`. Thứ tư, Streetwear đóng góp gần `79.9%` doanh thu nhưng cũng chiếm `79.6%` tổng tiền hoàn, còn COD có tỷ lệ huỷ đơn `16.0%`, cao gần gấp đôi thẻ tín dụng (`7.98%`). Dựa trên các insight này, nhóm đề xuất một bộ hành động ưu tiên gồm siết fixed promo, chuyển dịch COD sang prepaid, và giảm `wrong_size` trong Streetwear. Song song, nhóm xây dựng pipeline forecasting có kiểm soát leakage, cross-validation đúng horizon deploy và giải thích mô hình bằng SHAP.

## 4. Skeleton Báo Cáo

## 4.1 Introduction / Business Context

Mục tiêu kinh doanh của doanh nghiệp là tối ưu tăng trưởng doanh thu đồng thời kiểm soát lợi nhuận, tồn kho và chất lượng vận hành. Vì vậy, bài làm được chia thành hai phần bổ trợ nhau:

- `Phần 2`: tìm insight kinh doanh có thể hành động ngay
- `Phần 3`: xây dựng mô hình dự báo doanh thu phục vụ planning cho 18 tháng tiếp theo

Đoạn chốt nên nêu rõ: chúng tôi không chỉ tìm pattern mô tả, mà còn ưu tiên những insight có thể quy đổi thành quyết định vận hành, marketing hoặc merchandizing.

## 4.2 Data Scope

Có thể dùng đoạn ngắn sau:

> Dataset mô phỏng hoạt động của một nhà bán lẻ thời trang e-commerce Việt Nam trong giai đoạn `04/07/2012-31/12/2022`, gồm các lớp dữ liệu master, transaction, analytical và operational. Để tạo insight kinh doanh, chúng tôi kết hợp `orders`, `order_items`, `products`, `promotions`, `returns`, `web_traffic`, `inventory` và `sales`. Với bài toán forecasting, toàn bộ feature chỉ được tạo từ các file được cung cấp, không dùng dữ liệu ngoài.

## 4.3 Part 2: Visualizations and Analysis

### Insight 1. Demand reset sau 2018, nhưng seasonality vẫn rất mạnh

Figure đề xuất: `Demand baseline` từ `eda_round1.ipynb`.

Copy-ready talking points:

- Doanh thu ngày trung bình giảm từ khoảng `5.07M` trước 2019 xuống `3.01M` từ 2019 trở đi, tương đương giảm `40.6%`.
- Tuy vậy, seasonality vẫn ổn định: tháng 4 là peak season, doanh thu trung bình cao gần `2.9x` tháng 12.
- Theo ngày trong tuần, Wednesday cao hơn Saturday khoảng `19%`, gợi ý nhu cầu không hoàn toàn weekend-driven.

Business implication:

- Không nên forecast bằng toàn bộ lịch sử theo cùng một level.
- Planning cho inventory, campaign và manpower nên dồn vào cụm `Mar-Jun`, không phân bổ đều quanh năm.

### Insight 2. Orders và conversion quan trọng hơn raw traffic

Figure đề xuất: `Traffic vs orders` từ `eda_round1.ipynb`.

Copy-ready talking points:

- `corr(Revenue, Orders)=0.9377`, cao hơn nhiều so với `corr(Revenue, Sessions)=0.3211`.
- `conversion proxy = orders / visitors` có tương quan `0.6259` với Revenue, cao hơn raw sessions.
- Hàm ý là tăng traffic không tự động biến thành tăng doanh thu nếu conversion không đi cùng.

Business implication:

- Ưu tiên tối ưu funnel và merchandising hơn là mua thêm traffic đại trà.
- KPI nên theo dõi song song `orders`, `conversion`, `AOV`, thay vì chỉ nhìn sessions.

### Insight 3. Promotions tạo volume, nhưng fixed promos đang phá lợi nhuận

Figure đề xuất: `Promotion economics` từ `eda_round1.ipynb`.

Copy-ready talking points:

- Các dòng có promo vẫn đóng góp doanh thu lớn, nhưng margin thấp hơn đáng kể so với non-promo.
- Promo kiểu `percentage` vẫn giữ margin dương khoảng `6.0%`.
- Promo kiểu `fixed` tạo khoảng `376.5M VND` revenue nhưng gross profit âm khoảng `230.5M VND`, tương đương margin `-61.2%`.

Business implication:

- Không nên cắt toàn bộ promo; cần redesign promo mix.
- `fixed promotions` là ứng viên số 1 để audit điều kiện áp dụng, cap discount hoặc chuyển sang percentage tier.

### Insight 4. COD và wrong-size là hai leak doanh thu lớn nhất

Figure đề xuất: `Returns & cancellations` từ `eda_round1.ipynb`.

Copy-ready talking points:

- `COD` có tỷ lệ huỷ đơn `16.0%`, gần gấp đôi `credit_card` (`7.98%`).
- Nếu đưa COD về mức huỷ tương đương thẻ, doanh nghiệp có thể bảo toàn khoảng `197.5M VND` doanh thu.
- `wrong_size` chiếm khoảng `35%` lý do trả hàng.
- Riêng Streetwear chiếm `79.9%` doanh thu nhưng cũng chiếm `79.6%` tổng tiền hoàn.
- Chỉ cần giảm `10%` wrong-size refund của Streetwear đã tương đương tiết kiệm khoảng `14.0M VND`.

Business implication:

- Khuyến khích prepaid cho COD-risk cohorts là một quick win.
- Cần size guide, fit recommendation và quality control ưu tiên cho Streetwear trước tiên.

### Insight 5. Inventory đang bị lệch phân bổ, không đơn thuần là thiếu hàng

Figure đề xuất: `Category portfolio` từ `eda_round1.ipynb`.

Copy-ready talking points:

- Streetwear chiếm gần `80%` revenue nhưng vẫn có `stockout_flag` trung bình `67.3%` và `overstock_flag` trung bình `74.9%`.
- `days_of_supply` của Streetwear khoảng `887` ngày, còn Outdoor còn cao hơn ở mức `1068.8` ngày.
- Việc cùng lúc xuất hiện cả stockout lẫn overstock cho thấy bài toán nằm ở `SKU allocation / mix planning`, không chỉ ở mức tồn tổng.

Business implication:

- Cần tối ưu allocation theo category / SKU / season, thay vì tăng tồn kho đồng loạt.
- Streetwear nên là priority lane vì đây vừa là category lớn nhất, vừa là nguồn hoàn trả lớn nhất.

## 4.4 Part 3: Forecasting Pipeline

Nên viết ngắn, tập trung vào 4 ý:

1. Mục tiêu: forecast `Revenue` và `COGS` cho `01/01/2023-01/07/2024`.
2. Feature engineering: calendar, cyclic, seasonal profile, holiday/event flags.
3. Validation: `TimeSeriesSplit(n_splits=2, test_size=548)`, rebuild seasonal per fold, LOO seasonal cho train để tránh target encoding leakage.
4. Ensemble: LightGBM + Prophet, weights re-tuned dưới CV mới.

Đoạn mẫu:

> Chúng tôi dùng pipeline forecasting có kiểm soát leakage chặt chẽ. Toàn bộ seasonal features được build lại trong từng fold CV, và train rows dùng leave-one-year-out seasonal encoding để tránh hiện tượng mô hình nhìn thấy chính target của mình. Validation horizon được đặt bằng đúng deployment horizon (`548` ngày) thay vì dùng backtest ngắn hơn. Final ensemble kết hợp LightGBM với Prophet để tận dụng hai kiểu inductive bias khác nhau: tree model bắt event spikes tốt, còn Prophet ổn hơn cho long-term seasonality.

## 4.5 Validation and Results

Bạn có thể lấy trực tiếp từ `forecasting.ipynb`:

- Revenue CV: `R² ≈ 0.5685`, `MAPE ≈ 33.5%`
- COGS CV: `R² ≈ 0.6023`, `MAPE ≈ 38.1%`
- Tuned weights: Revenue `0.75 LGB + 0.25 Prophet`; COGS `0.90 LGB + 0.10 Prophet`

Điểm quan trọng nên nhấn:

- CV cũ đẹp hơn nhưng lạc quan do horizon ngắn và leakage-like seasonal encoding.
- CV hiện tại trung thực hơn với deployment.

## 4.6 Explainability

Figure đề xuất: `SHAP bar plot` và `SHAP beeswarm` từ `forecasting.ipynb`.

Copy-ready talking points:

- SHAP được dùng để xác định yếu tố dẫn động doanh thu theo ngôn ngữ business.
- Nếu `seasonal` và `holiday/event` features đứng đầu, mô hình đang học seasonality hợp lý thay vì bám nhiễu.
- Explainability không chỉ để trình bày mà còn để debug leakage / extrapolation.

## 4.7 Final Recommendations

Bạn có thể chốt report bằng 3 ý rất ngắn:

1. `Promotion redesign`: audit toàn bộ fixed promo trước, vì đây là nơi thấy thất thoát margin rõ nhất.
2. `Checkout and returns`: giảm COD-risk và xử lý wrong-size trong Streetwear trước, vì đây là leak doanh thu lớn nhất và dễ can thiệp.
3. `Inventory planning`: tái phân bổ tồn kho theo category-season mix, đặc biệt cho Streetwear và Outdoor.

## 4.8 Limitations

Nên ghi ngắn gọn:

- Không có exogenous feature thật sự cho giai đoạn test 2023-2024.
- Một số insight EDA mang tính tương quan hơn là nhân quả.
- Inventory là snapshot cuối tháng, nên không quan sát được luồng tồn kho hàng ngày.

## 5. Rubric Alignment Checklist

### Descriptive

- Có figure với title, axis labels, legend rõ ràng
- Có số cụ thể cho từng claim

### Diagnostic

- Có giải thích nguyên nhân khả dĩ: seasonality, promo mix, COD, size mismatch, inventory allocation

### Predictive

- Có phân tích seasonality / regime shift / leading indicators
- Có forecasting section với validation đúng chiều thời gian

### Prescriptive

- Có ít nhất 3 hành động cụ thể, ưu tiên theo giá trị kinh doanh
- Mỗi action gắn với một KPI có thể theo dõi

## 6. Gợi Ý KPI Để Gắn Với Từng Action

| Action | KPI chính | KPI phụ |
|---|---|---|
| Siết fixed promo | Gross margin per promo | Promo ROI, discount depth |
| Giảm COD risk | COD cancel rate | Prepaid adoption, checkout completion |
| Giảm wrong-size | Refund amount from wrong-size | Return rate theo size, review score |
| Tối ưu inventory allocation | Stockout flag + overstock flag | Days of supply, fill rate |
