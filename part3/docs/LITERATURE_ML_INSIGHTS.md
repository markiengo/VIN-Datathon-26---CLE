# Literature insights for ML forecasting (citable references)

Tài liệu tổng hợp **insight có ích cho phần machine learning** (tabular GBM, calibration, hierarchy, regime, holiday, validation) kèm **tên bài / nguồn** để team chèn vào report (References / Tài liệu tham khảo).  
Phân loại: **A** = bài học thuật quốc tế (journal / thesis / preprint); **B** = ứng dụng chuỗi thời gian tại Việt Nam (tạp chí trong nước / luận văn nước ngoài dữ liệu VN); **C** = so sánh mô hình e-com (bối cảnh tổng quát, không nhất thiết VN).

**Lưu ý phương pháp:** Kết quả “mô hình nào thắng” **phụ thuộc tần suất, độ dài horizon, và regime** — khi viết report nên trích dẫn **nguyên tắc** (break, holiday, reconciliation) hơn là copy kết luận “LSTM luôn tốt nhất” từ một dataset khác.

---

## 1. Insight cho ML (ánh xạ trực tiếp tới pipeline của team)

### 1.1 Regime change & marketing / elasticity (tabular + GBM)

- **Insight:** Khi có **structural break** (tương tự 2018→2019 trong `sales.csv`), hệ số tác động của giá/khuyến mãi (hoặc proxy) **không nên cố định** một regime; mô hình chỉ “scale toàn cục” có thể vẫn lệch shape ngày.
- **Ứng dụng ML:** Với LightGBM/XGBoost — thử **feature tương tác năm × promo**, **segment theo regime** (pre/post), hoặc **hai mô hình con** rồi blend theo ngày; **validation** phải có **cửa sổ qua break** (khớp `MODEL_RESEARCH_RETROSPECTIVE.md`).
- **Reference:** Huang, Fildes, & Soopramanien (2019) — *Forecasting retailer product sales in the presence of structural change* (*European Journal of Operational Research*).

### 1.2 Bất định nhiều break trên horizon dài (~548 ngày)

- **Insight:** Trên horizon dài, **một break đã ước lượng** có thể không đủ; văn hướng dẫn trung bình dự báo qua **bất định về break tương lai**.
- **Ứng dụng ML:** Thay vì chỉnh một lần `is_post_2018`, có thể thử **ensemble nhiều checkpoint train** (multi-origin backtest) hoặc **bagging theo thời điểm cắt train** để mô phỏng break uncertainty (heuristic gần tinh thần literature).
- **Reference:** Pesaran, Pettenuzzo, & Timmermann (2006) — *Forecasting time series subject to multiple structural breaks* (*The Review of Economic Studies*).  
- **Bổ sung (Bayesian break):** van den Hauwe, Paap, & van Dijk (2011) — *An alternative Bayesian approach to structural breaks in time series models* (Tinbergen Institute Discussion Paper).

### 1.3 Micro → aggregate (COGS / Revenue ngày)

- **Insight:** Dự báo cấp dưới (SKU, category, order-mix) cần **reconcile** về tổng ngày để tránh mâu thuẫn và khai thác tín hiệu bottom-up có kiểm soát.
- **Ứng dụng ML:** Khớp với nhánh **iter-38/39** (micro simulator + hierarchical blend): dùng **MinT / optimal reconciliation** làm lớp sau cùng các dự báo GBM theo nhóm (nếu team có vector phụ đủ dài) thay vì chỉ blend scalar cố định.
- **Reference:** Wickramasuriya, Athanasopoulos, & Hyndman (2019) — *Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization* (*Journal of the American Statistical Association*).

### 1.4 Sau reconcile: giữ mùa vụ / spike (Tết, 11/11)

- **Insight:** Reconcile thuần có thể **làm phẳng** biến động theo nhóm ngày lễ; cần **bước chỉnh có cấu trúc** sau cùng nếu dùng hierarchy sâu.
- **Ứng dụng ML:** Nếu áp reconciliation — giữ **regressor lễ** ở tầng top hoặc **post-adjust** theo stratum ngày lễ (tinh thần multi-stage).
- **Reference:** Yang et al. (2024) — *A Comprehensive Forecasting Framework based on Multi-Stage Hierarchical Forecasting Reconciliation and Adjustment (Multi-Stage HiFoReAd)* (arXiv:2412.14718).

### 1.5 Lễ di chuyển (Tết) — đặc trưng tabular thay vì chỉ Fourier dương lịch

- **Insight:** Lễ **lịch âm / cửa sổ nhiều ngày** nên mô hình hóa bằng **regressor theo lịch** + phần dư có cấu trúc (ARIMA / state space) hơn là chỉ sin/cos tháng dương.
- **Ứng dụng ML:** Đã có `days_from_tet` trong notebook — literature củng cố thêm **pha pre/core/post**, **tương tác cuối tuần × Tết**, và kiểm tra **OOS** trên các năm Tết khác nhau.
- **References:** Bell & Hillmer (1983) — *Modeling time series with calendar variation* (*JASA*). Lin & Liu — *Modeling Lunar Calendar Holiday Effects in Taiwan* (U.S. Census Bureau / ADRM working paper; analogue cho Tết). Zhang, Huang, & Yang (2020) — *A sales forecasting model for the consumer goods with holiday effects* (*Journal of Risk Analysis and Crisis Response*).

### 1.6 Baseline thống kê vs GBM / LSTM — không bỏ qua khi viết methodology

- **Insight (VN e-com, weekly):** Trên dữ liệu một công ty TMĐT Việt Nam, **SARIMAX** (+ ngoại sinh khuyến mãi) có thể **thắt chặt** Prophet/LSTM về MAPE tuần — nhắc nhở **độ phức tạp không đồng nghĩa tốt hơn** và cần **lịch khuyến mãi** làm ngoại sinh.
- **Ứng dụng ML:** Giữ **SARIMAX/ETS** (hoặc `sktime`) làm **baseline có kiểm chứng** trong report; so sánh với two-stage LGB/XGB trên **cùng** backtest hai cửa sổ 548 ngày.
- **Reference:** Pham (2024) — *A comparative analysis of demand forecasting models: A case study of a Vietnam e-commerce company* (Master’s thesis, Aalto University — Aaltodoc).

### 1.7 So sánh LSTM / SARIMA / XGBoost trên e-com (bối cảnh tổng quát)

- **Insight:** Có bài so sánh ba họ máy trên dữ liệu e-com thực — kết quả **phụ thuộc paper/dataset**; dùng để **biện minh** việc team chọn **gradient boosting + calibration** thay vì LSTM dài hạn, hoặc để **thử hybrid** (residual của SARIMA đưa vào GBM) nếu còn quota thử nghiệm.
- **References:** *E-Commerce Sales Forecasting by Comparing LSTM, SARIMA, and XGBoost Models* (*Journal of Emerging Technology and Digital Transformation* — bài trên nền tảng JETDT, xem trang bài để lấy năm/tập/chính xác). Ni, Huang, & Fu (2025) — *A stacking-based fusion framework for dynamic demand forecasting in e-commerce* (*Mathematics*, 13(21), 3436, DOI: 10.3390/math13213436).

### 1.8 Dự báo lô hàng khuyến mãi (ARIMA vs LSTM)

- **Insight:** Bài so sánh **shipment khuyến mãi** cho merchant e-com: cả ARIMA và LSTM đều có giá trị; gợi ý **feature chiến dịch** và chuỗi ngắn cho tầng vận hành.
- **Ứng dụng ML:** Liên hệ nhánh **promo / inventory** đã thử (iter-45): dùng làm **supporting citation** cho “promo-aware” demand, không thay thế validation.
- **Reference:** *A Study on Promotional Shipment Forecasting for E-Commerce Merchants Based on ARIMA Time Series and LSTM Models* (*Highlights in Science, Engineering and Technology* — tra trang bài cho metadata đầy đủ).

### 1.9 Chuỗi cung ứng Việt Nam (hybrid ARIMAX–LSTM)

- **Insight:** Case **cà phê** VN trên *Journal of Forecasting*: hybrid **ARIMAX–LSTM** cải thiện so với mô hình đơn — ý tưởng **tách tuyến tính / phi tuyến** (residual learning) tương thích với **stacking / GBM trên residual SARIMA**.
- **Reference:** Nguyen, T. T. H., Bekrar, A., Le, T. M., Abed, M., & Kantasa-ard, A. (2025). *Toward a smart forecasting model in supply chain management: A case study of coffee in Vietnam.* *Journal of Forecasting*, 44(1), 173–199. `https://doi.org/10.1002/for.3189`

### 1.10 Ứng dụng ARIMA/SARIMA tại Việt Nam (chuỗi tháng / xuất khẩu — analog method)

- **Insight:** Nhiều bài trong nước dùng **Box–Jenkins** cho chuỗi **vĩ mô / xuất khẩu / du lịch**; có thể trích để **biện minh** quy trình kiểm định dư, chọn bậc ARIMA, và so sánh với ML — **không** khẳng định cùng phân phối với daily SKU TMĐT.
- **References (B — ví dụ có URL công khai):**  
  - *Dự báo kim ngạch xuất khẩu hàng may mặc của Việt Nam bằng mô hình chuỗi thời gian ARIMA* — *Tạp chí Quản lý nhà nước* (2024).  
  - *Vận dụng mô hình ARIMA dự báo lượt khách quốc tế đến Việt Nam* — *Tạp chí Quản lý nhà nước* (2025).  
  - *Ứng dụng mô hình Sarima dự báo sản lượng xuất khẩu bột cá của Việt Nam* — *Tạp chí Khoa học Đại học Cần Thơ* (article view: `https://ctujsvn.ctu.edu.vn/index.php/ctujsvn/article/view/5247`).

### 1.11 XGBoost + marketing / nền tảng e-com (tổng quát)

- **Insight:** Bài gần đây về **XGBoost** + chiến lược dữ liệu marketing cho nền tảng e-com — hỗ trợ mục **feature từ hành vi / kênh** nếu có trong tương lai (trong đề public hiện tại thì hạn chế exogenous sau 2022).
- **Reference:** *Sales Forecasting and Data-Driven Marketing Strategies for E-Commerce Platforms Using XGBoost* (*International Journal of Information Technology and Web Engineering* / IGI Global, 2025 — tra IDEAS: `https://ideas.repec.org/a/igg/jiit00/v21y2025i1p1-21.html`).

---

## 2. Cách dùng trong report (gợi ý ngắn)

- **Mục Methodology / Model:** trích **Pham (2024)** + **Huang et al. (2019)** + **Bell & Hillmer (1983)** / **Lin & Liu** cho **regime + holiday + baseline SARIMAX**.  
- **Mục Hierarchical / micro:** **Wickramasuriya et al. (2019)** + **Yang et al. (2024)**.  
- **Mục So sánh ML:** **Ni et al. (2025)** + bài **JETDT (LSTM/SARIMA/XGBoost)** + **coffee Vietnam** cho **hybrid / stacking**.  
- **Mục Bối cảnh VN (không trùng daily TMĐT):** các bài **ARIMA/SARIMA** *Quản lý nhà nước* / *CTU* — ghi rõ “**chuỗi vĩ mô / tháng**, analog phương pháp”.

---

## 3. Danh mục trích dẫn (copy-paste cho References)

*(Chỉnh sửa DOI/năm/tập theo bản PDF chính thức bạn tải về.)*

1. Bell, W. R., & Hillmer, S. C. (1983). Modeling time series with calendar variation. *Journal of the American Statistical Association*, 78(383), 526–534. `https://doi.org/10.1080/01621459.1983.10478005`

2. Huang, T., Fildes, R., & Soopramanien, D. (2019). Forecasting retailer product sales in the presence of structural change. *European Journal of Operational Research*, 279(2), 459–470. `https://doi.org/10.1016/j.ejor.2019.06.011`

3. Pesaran, M. H., Pettenuzzo, D., & Timmermann, A. (2006). Forecasting time series subject to multiple structural breaks. *The Review of Economic Studies*, 73(4), 1057–1084.

4. van den Hauwe, S., Paap, R., & van Dijk, D. J. C. (2011). An alternative Bayesian approach to structural breaks in time series models. Tinbergen Institute Discussion Paper TI 2011-023/4.

5. Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *Journal of the American Statistical Association*. `https://doi.org/10.1080/01621459.2018.1448825`

6. Yang, Z., et al. (2024). A Comprehensive Forecasting Framework based on Multi-Stage Hierarchical Forecasting Reconciliation and Adjustment (Multi-Stage HiFoReAd). arXiv:2412.14718. `https://arxiv.org/abs/2412.14718`

7. Zhang, M., Huang, X.-n., & Yang, C.-b. (2020). A sales forecasting model for the consumer goods with holiday effects. *Journal of Risk Analysis and Crisis Response*, 10(2), 69–76. `https://doi.org/10.2991/jracr.k.200709.001`

8. Ni, L., Huang, Z., & Fu, N. (2025). A stacking-based fusion framework for dynamic demand forecasting in e-commerce. *Mathematics*, 13(21), 3436. `https://doi.org/10.3390/math13213436`

9. Pham, T. (2024). *A comparative analysis of demand forecasting models: A case study of a Vietnam e-commerce company* [Master’s thesis, Aalto University]. Aaltodoc. `https://aaltodoc.aalto.fi/items/67a8570b-d167-4daf-abf5-5d000fbf78e9`

10. *E-Commerce Sales Forecasting by Comparing LSTM, SARIMA, and XGBoost Models.* *Journal of Emerging Technology and Digital Transformation*. Article view: `https://journalofemergingtechnologyanddigitaltransformation.com/index.php/3/article/view/41`

11. *A Study on Promotional Shipment Forecasting for E-Commerce Merchants Based on ARIMA Time Series and LSTM Models.* *Highlights in Science, Engineering and Technology*. Article view: `https://drpress.org/ojs/index.php/HSET/article/view/21255`

12. Nguyen, T. T. H., Bekrar, A., Le, T. M., Abed, M., & Kantasa-ard, A. (2025). Toward a smart forecasting model in supply chain management: A case study of coffee in Vietnam. *Journal of Forecasting*, 44(1), 173–199. `https://doi.org/10.1002/for.3189`

13. [Sales forecasting XGBoost e-commerce platforms — IGI / IJITWE]. `https://ideas.repec.org/a/igg/jiit00/v21y2025i1p1-21.html`

14. Lin, J.-L., & Liu, T.-S. *Modeling Lunar Calendar Holiday Effects in Taiwan* (U.S. Census Bureau / ADRM; working paper — lấy đúng tiêu đề & năm từ trang IDEAS/RePEc hoặc Census khi trích dẫn).

15. *Dự báo kim ngạch xuất khẩu hàng may mặc của Việt Nam bằng mô hình chuỗi thời gian ARIMA.* (2024). *Tạp chí Quản lý nhà nước*. `https://www.quanlynhanuoc.vn/2024/11/07/du-bao-kim-ngach-xuat-khau-hang-may-mac-cua-viet-nam-bang-moi-hinh-chuoi-thoi-gian-arima/`

16. *Vận dụng mô hình ARIMA dự báo lượt khách quốc tế đến Việt Nam.* (2025). *Tạp chí Quản lý nhà nước*. `https://www.quanlynhanuoc.vn/2025/10/03/van-dung-moi-hinh-arima-du-bao-luot-khach-quoc-te-den-viet-nam/`

17. *Ứng dụng mô hình Sarima dự báo sản lượng xuất khẩu bột cá của Việt Nam.* *Tạp chí Khoa học Đại học Cần Thơ*. `https://ctujsvn.ctu.edu.vn/index.php/ctujsvn/article/view/5247`

---

## 4. Liên kết nội bộ

- [`MODEL_RESEARCH_RETROSPECTIVE.md`](./MODEL_RESEARCH_RETROSPECTIVE.md) — thí nghiệm đã chạy trong cuộc thi.  
- [`MODEL_CARD.md`](../MODEL_CARD.md) — định nghĩa pipeline reproducible vs submission iter-39.

---

*Tổng hợp từ vòng literature (quốc tế + Scholar/từ khóa tiếng Việt). Khi in report, nên mở từng URL/DOI và điền đủ tác giả, tập, số, trang theo chuẩn BTC yêu cầu (APA / Harvard / GB).*
