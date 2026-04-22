import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).parent.parent   # project root (scripts/ is one level below)


def to_source(text: str):
    text = dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]


def md_cell(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source(text),
    }


def code_cell(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(text),
    }


cells = [
    md_cell(
        """
        # Phần 2 — Notebook EDA Vòng 1

        Notebook này được thiết kế để kéo điểm **Phần 2 — Trực quan hoá & Phân tích Dữ liệu** theo đúng rubric vòng 1:

        - **Mô tả**: mô tả đúng mẫu hình và quy mô ảnh hưởng
        - **Chẩn đoán**: giải thích nguyên nhân khả dĩ
        - **Dự báo**: chỉ ra tính mùa vụ và các chỉ báo sớm
        - **Khuyến nghị**: chốt thành hành động có KPI và mức độ ưu tiên

        Storyline chính của notebook:

        1. Nhu cầu bước vào trạng thái mới từ 2019 nhưng mùa vụ vẫn rất rõ
        2. Doanh thu đi cùng **số đơn** và **chuyển đổi** nhiều hơn lưu lượng truy cập thô
        3. Khuyến mãi có tác dụng, nhưng một số cơ chế đang phá biên lợi nhuận
        4. COD, wrong-size và phân bổ tồn kho là các điểm rò rỉ hiệu quả lớn nhất
        """
    ),
    code_cell(
        """
        import warnings
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        warnings.filterwarnings("ignore")

        plt.rcParams.update(
            {
                "figure.figsize": (12, 5),
                "figure.dpi": 130,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
            }
        )

        DATA = Path("dataset")

        sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"])
        orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
        order_items = pd.read_csv(DATA / "order_items.csv", low_memory=False)
        products = pd.read_csv(DATA / "products.csv")
        promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
        returns = pd.read_csv(DATA / "returns.csv", parse_dates=["return_date"])
        geography = pd.read_csv(DATA / "geography.csv")
        inventory = pd.read_csv(DATA / "inventory.csv", parse_dates=["snapshot_date"])
        web = pd.read_csv(DATA / "web_traffic.csv", parse_dates=["date"])

        def fmt_billion(x, _):
            return f"{x / 1e9:.1f}B"

        def fmt_million(x, _):
            return f"{x / 1e6:.0f}M"

        BILLION_FMT = FuncFormatter(fmt_billion)
        MILLION_FMT = FuncFormatter(fmt_million)

        print("Đã tải các bảng dữ liệu:")
        print(
            {
                "sales": sales.shape,
                "orders": orders.shape,
                "order_items": order_items.shape,
                "products": products.shape,
                "promotions": promotions.shape,
                "returns": returns.shape,
                "inventory": inventory.shape,
                "web_traffic": web.shape,
            }
        )
        """
    ),
    md_cell(
        """
        ## Tóm tắt điều hành

        Bốn phát hiện chính mà notebook này muốn chứng minh:

        - Doanh thu ngày trung bình giảm mạnh sau 2018, nhưng **hình dạng mùa vụ** vẫn ổn định và đủ mạnh để dùng cho lập kế hoạch.
        - Doanh thu bám theo **số đơn** và **chuyển đổi** mạnh hơn hẳn so với lưu lượng truy cập thô, nên bài toán không chỉ là "đổ thêm lượt truy cập".
        - Khuyến mãi kiểu `percentage` vẫn dùng được, nhưng khuyến mãi kiểu `fixed` đang tạo doanh thu với **lợi nhuận gộp âm**.
        - Streetwear là động cơ doanh thu lớn nhất, nhưng cũng là điểm tập trung của hoàn tiền, trả hàng và lệch pha tồn kho.
        """
    ),
    code_cell(
        """
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_vi = {
            "Monday": "Thứ 2",
            "Tuesday": "Thứ 3",
            "Wednesday": "Thứ 4",
            "Thursday": "Thứ 5",
            "Friday": "Thứ 6",
            "Saturday": "Thứ 7",
            "Sunday": "Chủ nhật",
        }
        month_labels = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"]

        items = (
            order_items.merge(
                orders[
                    [
                        "order_id",
                        "order_date",
                        "payment_method",
                        "order_source",
                        "device_type",
                        "order_status",
                        "zip",
                    ]
                ],
                on="order_id",
                how="left",
            )
            .merge(products[["product_id", "category", "segment", "size", "price", "cogs"]], on="product_id", how="left")
            .merge(promotions[["promo_id", "promo_type", "promo_channel"]], on="promo_id", how="left")
            .merge(geography[["zip", "region", "city"]], on="zip", how="left")
        )
        items["revenue"] = items["unit_price"] * items["quantity"]
        items["gross_profit"] = items["revenue"] - items["cogs"] * items["quantity"]
        items["discount_rate"] = items["discount_amount"] / (items["revenue"] + items["discount_amount"]).replace(0, np.nan)
        items["promo_used"] = items["promo_id"].notna()
        items["year"] = items["order_date"].dt.year
        items["month"] = items["order_date"].dt.month

        order_revenue = items.groupby("order_id")["revenue"].sum().rename("order_revenue")
        order_gp = items.groupby("order_id")["gross_profit"].sum().rename("order_gp")
        returned = returns.groupby("order_id").agg(
            return_lines=("return_id", "count"),
            refund_amount=("refund_amount", "sum"),
        )
        order_level = (
            orders.merge(order_revenue, left_on="order_id", right_index=True, how="left")
            .merge(order_gp, left_on="order_id", right_index=True, how="left")
            .merge(returned, left_on="order_id", right_index=True, how="left")
        )
        order_level[["order_revenue", "order_gp", "return_lines", "refund_amount"]] = order_level[
            ["order_revenue", "order_gp", "return_lines", "refund_amount"]
        ].fillna(0)
        order_level["cancelled"] = order_level["order_status"].eq("cancelled")
        order_level["returned"] = order_level["return_lines"].gt(0)

        daily_web = (
            web.groupby("date")
            .agg(
                sessions=("sessions", "sum"),
                visitors=("unique_visitors", "sum"),
                page_views=("page_views", "sum"),
            )
            .reset_index()
        )
        daily_orders = orders.groupby("order_date").size().rename("orders").reset_index()
        daily = (
            sales.merge(daily_web, left_on="Date", right_on="date", how="left")
            .merge(daily_orders, left_on="Date", right_on="order_date", how="left")
        )
        daily["orders"] = daily["orders"].fillna(0)
        daily["conversion_proxy"] = daily["orders"] / daily["visitors"]
        daily["aov"] = daily["Revenue"] / daily["orders"].replace(0, np.nan)
        daily["year"] = daily["Date"].dt.year
        daily["month"] = daily["Date"].dt.month

        recent = sales.loc[sales["Date"] >= "2019-01-01"].copy()
        recent["year"] = recent["Date"].dt.year
        recent["month"] = recent["Date"].dt.month
        recent["dow"] = pd.Categorical(recent["Date"].dt.day_name(), categories=weekday_order, ordered=True)

        annual_avg = sales.assign(year=sales["Date"].dt.year).groupby("year")[["Revenue", "COGS"]].mean().reset_index()
        monthly_recent = recent.groupby("month")[["Revenue", "COGS"]].mean().reindex(range(1, 13))
        dow_recent = recent.groupby("dow")[["Revenue", "COGS"]].mean().reindex(weekday_order)

        returns_enriched = returns.merge(products[["product_id", "category", "segment", "size"]], on="product_id", how="left")
        return_reason = returns_enriched["return_reason"].value_counts(normalize=True).mul(100)
        size_return_rate = (
            returns_enriched.groupby("size").size() / items.groupby("size").size()
        ).reindex(["S", "M", "L", "XL"]).mul(100)
        category_return_rate = (
            returns_enriched.groupby("category").size() / items.groupby("category").size()
        ).sort_values(ascending=False).mul(100)

        promo_summary = items.groupby("promo_used").agg(
            lines=("order_id", "size"),
            revenue=("revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            avg_discount_rate=("discount_rate", "mean"),
        )
        promo_summary["margin"] = promo_summary["gross_profit"] / promo_summary["revenue"]
        promo_summary.index = promo_summary.index.map({False: "Không dùng khuyến mãi", True: "Có dùng khuyến mãi"})

        promo_type = items.loc[items["promo_used"]].groupby("promo_type").agg(
            lines=("order_id", "size"),
            revenue=("revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            avg_discount_rate=("discount_rate", "mean"),
        )
        promo_type["margin"] = promo_type["gross_profit"] / promo_type["revenue"]

        promo_channel = items.loc[items["promo_used"]].groupby("promo_channel").agg(
            lines=("order_id", "size"),
            revenue=("revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
        promo_channel["margin"] = promo_channel["gross_profit"] / promo_channel["revenue"]
        promo_channel = promo_channel.sort_values("revenue", ascending=False)

        payment_cancel = order_level.groupby("payment_method")["cancelled"].mean().mul(100).sort_values(ascending=False)

        category_portfolio = items.groupby("category").agg(
            revenue=("revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            lines=("order_id", "size"),
        )
        category_portfolio["rev_share"] = category_portfolio["revenue"] / category_portfolio["revenue"].sum()
        category_portfolio["margin"] = category_portfolio["gross_profit"] / category_portfolio["revenue"]
        category_portfolio = category_portfolio.join(
            (
                returns_enriched.groupby("category").size()
                / items.groupby("category").size()
            ).rename("return_rate")
        )
        category_portfolio = category_portfolio.join(
            inventory.groupby("category")[["stockout_flag", "overstock_flag", "days_of_supply", "fill_rate"]].mean()
        )
        category_portfolio = category_portfolio.sort_values("revenue", ascending=False)

        rates = order_level.groupby("payment_method")["cancelled"].mean()
        cod_orders = order_level.loc[order_level["payment_method"] == "cod"]
        cod_revenue_at_risk = max(rates.get("cod", 0) - rates.get("credit_card", 0), 0) * len(cod_orders) * cod_orders.loc[
            cod_orders["cancelled"], "order_revenue"
        ].mean()
        fixed_gp_drag = abs(float(promo_type.loc["fixed", "gross_profit"])) if "fixed" in promo_type.index else np.nan
        streetwear_wrong_size_refund = returns_enriched.loc[
            (returns_enriched["category"] == "Streetwear") & (returns_enriched["return_reason"] == "wrong_size"),
            "refund_amount",
        ].sum()

        action_df = pd.DataFrame(
            [
                {
                    "initiative": "Thiết kế lại hoặc dừng khuyến mãi giảm tiền cố định",
                    "annual_value_vnd": fixed_gp_drag,
                    "kpi": "Biên lợi nhuận gộp theo khuyến mãi",
                    "why_now": "Khuyến mãi giảm thẳng tiền hiện đang cho lợi nhuận gộp âm.",
                },
                {
                    "initiative": "Kéo tỷ lệ huỷ COD về gần mức thẻ",
                    "annual_value_vnd": cod_revenue_at_risk,
                    "kpi": "Tỷ lệ huỷ COD",
                    "why_now": "COD đang có tỷ lệ huỷ gần gấp đôi nhóm thẻ.",
                },
                {
                    "initiative": "Giảm 10% hoàn tiền do sai size của Streetwear",
                    "annual_value_vnd": streetwear_wrong_size_refund * 0.10,
                    "kpi": "Giá trị hoàn tiền do sai size",
                    "why_now": "Streetwear tập trung cả quy mô lẫn nỗi đau về trả hàng.",
                },
            ]
        ).sort_values("annual_value_vnd", ascending=True)

        print("Đã chuẩn bị xong các bảng phân tích.")
        """
    ),
    md_cell(
        """
        ## 1. Nền cầu cơ bản: bước ngoặt cấu trúc và tính mùa vụ

        Phần này trả lời ba câu hỏi:

        - **Điều gì đã xảy ra?** Mức cầu thay đổi thế nào theo năm?
        - **Nhu cầu đạt đỉnh khi nào?** Mùa nào và ngày nào trong tuần quan trọng nhất?
        - **Nên làm gì?** Kế hoạch tồn kho, chiến dịch và nhân lực nên dồn vào đâu?
        """
    ),
    code_cell(
        """
        pre_2019_avg = sales.loc[sales["Date"] <= "2018-12-31", "Revenue"].mean()
        post_2019_avg = sales.loc[sales["Date"] >= "2019-01-01", "Revenue"].mean()
        drop_pct = (1 - post_2019_avg / pre_2019_avg) * 100

        peak_month = int(monthly_recent["Revenue"].idxmax())
        trough_month = int(monthly_recent["Revenue"].idxmin())
        peak_dow = dow_recent["Revenue"].idxmax()
        trough_dow = dow_recent["Revenue"].idxmin()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        year_colors = ["#b8c0cc" if y < 2019 else "#006d77" for y in annual_avg["year"]]
        axes[0].bar(annual_avg["year"].astype(str), annual_avg["Revenue"], color=year_colors)
        axes[0].set_title("Doanh thu ngày trung bình theo năm")
        axes[0].set_xlabel("Năm")
        axes[0].set_ylabel("Doanh thu")
        axes[0].yaxis.set_major_formatter(BILLION_FMT)
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].plot(month_labels, monthly_recent["Revenue"].values, marker="o", color="#006d77")
        axes[1].set_title("Doanh thu ngày trung bình theo tháng (2019-2022)")
        axes[1].set_xlabel("Tháng")
        axes[1].set_ylabel("Doanh thu")
        axes[1].yaxis.set_major_formatter(MILLION_FMT)

        axes[2].bar([weekday_vi[d] for d in dow_recent.index.astype(str)], dow_recent["Revenue"].values, color="#83c5be")
        axes[2].set_title("Doanh thu ngày trung bình theo ngày trong tuần (2019-2022)")
        axes[2].set_xlabel("Ngày trong tuần")
        axes[2].set_ylabel("Doanh thu")
        axes[2].yaxis.set_major_formatter(MILLION_FMT)
        axes[2].tick_params(axis="x", rotation=35)

        plt.tight_layout()
        plt.show()

        print(f"Doanh thu ngày trung bình giảm từ {pre_2019_avg:,.0f} xuống {post_2019_avg:,.0f} VND sau năm 2018 (giảm {drop_pct:.1f}%).")
        print(
            f"Tháng đỉnh = {month_labels[peak_month - 1]} với {monthly_recent.loc[peak_month, 'Revenue']:,.0f} VND/ngày; "
            f"tháng thấp nhất = {month_labels[trough_month - 1]} với {monthly_recent.loc[trough_month, 'Revenue']:,.0f} VND/ngày."
        )
        print(
            f"Ngày mạnh nhất = {weekday_vi[peak_dow]} với {dow_recent.loc[peak_dow, 'Revenue']:,.0f} VND/ngày; "
            f"ngày yếu nhất = {weekday_vi[trough_dow]} với {dow_recent.loc[trough_dow, 'Revenue']:,.0f} VND/ngày."
        )
        """
    ),
    md_cell(
        """
        **Diễn giải**

        - Đây không còn là một chuỗi "một trạng thái" xuyên suốt 2012-2022. Nếu huấn luyện mô hình hoặc lập kế hoạch mà đối xử toàn bộ lịch sử như nhau, bài toán sẽ bị lệch về mức doanh thu cũ.
        - Mùa vụ vẫn rất rõ ngay cả sau khi mặt bằng doanh thu đã đổi. Điều này tạo nền tảng tốt cho cả lập kế hoạch và dự báo.

        **Hàm ý kinh doanh**

        - Nên ưu tiên hàng, ngân sách chiến dịch và năng lực giao vận cho cụm `Mar-Jun`.
        - Không nên đánh giá hiệu quả tháng 11-12 theo cùng kỳ vọng mặt bằng với tháng 3-6.
        """
    ),
    md_cell(
        """
        ## 2. Lượt truy cập không phải toàn bộ câu chuyện: số đơn và chuyển đổi mới là đòn bẩy gần doanh thu hơn

        Phần này nhằm tránh một bẫy phổ biến trong EDA thương mại điện tử: nhìn thấy lượt truy cập tăng rồi mặc định doanh thu sẽ tăng tương ứng.
        """
    ),
    code_cell(
        """
        monthly_flow = daily.set_index("Date").resample("ME").sum(numeric_only=True)[["Revenue", "orders", "sessions"]]
        indexed_flow = monthly_flow / monthly_flow.iloc[0] * 100

        corr_table = daily[["Revenue", "orders", "sessions", "visitors", "conversion_proxy", "aov"]].corr()["Revenue"].sort_values(ascending=False)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(indexed_flow.index, indexed_flow["Revenue"], label="Doanh thu", lw=2, color="#264653")
        axes[0].plot(indexed_flow.index, indexed_flow["orders"], label="Số đơn", lw=2, color="#2a9d8f")
        axes[0].plot(indexed_flow.index, indexed_flow["sessions"], label="Phiên truy cập", lw=2, color="#e9c46a")
        axes[0].set_title("Diễn biến chỉ số theo tháng (gốc = tháng đầu = 100)")
        axes[0].set_xlabel("Tháng")
        axes[0].set_ylabel("Chỉ số")
        axes[0].legend()

        axes[1].scatter(daily["orders"], daily["Revenue"], alpha=0.25, s=14, color="#2a9d8f")
        axes[1].set_title(f"Doanh thu và số đơn (corr = {corr_table['orders']:.2f})")
        axes[1].set_xlabel("Số đơn mỗi ngày")
        axes[1].set_ylabel("Doanh thu")
        axes[1].yaxis.set_major_formatter(BILLION_FMT)

        axes[2].scatter(daily["sessions"], daily["Revenue"], alpha=0.25, s=14, color="#f4a261")
        axes[2].set_title(f"Doanh thu và phiên truy cập (corr = {corr_table['sessions']:.2f})")
        axes[2].set_xlabel("Phiên truy cập mỗi ngày")
        axes[2].set_ylabel("Doanh thu")
        axes[2].yaxis.set_major_formatter(BILLION_FMT)

        plt.tight_layout()
        plt.show()

        print("Tương quan với doanh thu")
        print(corr_table.round(4).to_string())
        """
    ),
    md_cell(
        """
        **Diễn giải**

        - Doanh thu đi cùng **số đơn** gần như trực tiếp, trong khi sessions thô yếu hơn nhiều.
        - `conversion_proxy = orders / visitors` mang nhiều tín hiệu hơn lưu lượng truy cập thô, nghĩa là hiệu quả phễu chuyển đổi quan trọng hơn chuyện chỉ tăng đầu phễu.

        **Hàm ý kinh doanh**

        - Nếu ngân sách có hạn, ưu tiên tối ưu tỷ lệ chuyển đổi, chất lượng trang đích, cơ cấu sản phẩm và trưng bày hàng hóa trước khi mua thêm lượt truy cập.
        - Trong bảng theo dõi vận hành, nên theo dõi `orders`, `conversion`, `AOV` cùng với sessions thay vì xem sessions một mình.
        """
    ),
    md_cell(
        """
        ## 3. Kinh tế học khuyến mãi: có doanh thu kéo lên, nhưng biên lợi nhuận không phải lúc nào cũng đi cùng

        Phần này chuyển phân tích từ "khuyến mãi có dùng nhiều không?" sang câu hỏi mà doanh nghiệp thực sự quan tâm:

        - Khuyến mãi nào giữ được lợi nhuận gộp?
        - Khuyến mãi nào chỉ tạo doanh thu bề mặt?
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].bar(promo_summary.index, promo_summary["revenue"], color=["#adb5bd", "#457b9d"])
        axes[0].set_title("Doanh thu theo trạng thái dùng khuyến mãi")
        axes[0].set_xlabel("Trạng thái khuyến mãi")
        axes[0].set_ylabel("Doanh thu")
        axes[0].yaxis.set_major_formatter(BILLION_FMT)

        promo_margin_pct = promo_type["margin"].mul(100).sort_values(ascending=False)
        axes[1].bar(promo_margin_pct.index.astype(str), promo_margin_pct.values, color=["#2a9d8f", "#e76f51"])
        axes[1].axhline(0, color="black", lw=1)
        axes[1].set_title("Biên lợi nhuận gộp theo loại khuyến mãi")
        axes[1].set_xlabel("Loại khuyến mãi")
        axes[1].set_ylabel("Biên lợi nhuận gộp (%)")
        for idx, value in enumerate(promo_margin_pct.values):
            axes[1].text(idx, value + (1 if value >= 0 else -3), f"{value:.1f}%", ha="center")

        top_channels = promo_channel.head(5).sort_values("revenue", ascending=True)
        axes[2].barh(top_channels.index.astype(str), top_channels["revenue"], color="#8ecae6")
        axes[2].set_title("Top kênh khuyến mãi theo doanh thu")
        axes[2].set_xlabel("Doanh thu")
        axes[2].set_ylabel("Kênh khuyến mãi")
        axes[2].xaxis.set_major_formatter(BILLION_FMT)

        plt.tight_layout()
        plt.show()

        print("Tóm tắt theo trạng thái dùng khuyến mãi")
        print((promo_summary.assign(margin=promo_summary["margin"] * 100, avg_discount_rate=promo_summary["avg_discount_rate"] * 100)).round(2).to_string())
        print()
        print("Tóm tắt theo loại khuyến mãi")
        print((promo_type.assign(margin=promo_type["margin"] * 100, avg_discount_rate=promo_type["avg_discount_rate"] * 100)).round(2).to_string())
        """
    ),
    md_cell(
        """
        **Diễn giải**

        - Khuyến mãi vẫn là phần quan trọng của cỗ máy thương mại, nhưng không phải loại nào cũng "lành mạnh".
        - Khuyến mãi kiểu `percentage` vẫn giữ được biên dương.
        - Khuyến mãi kiểu `fixed` là tín hiệu cảnh báo rất mạnh vì đang ăn mòn lợi nhuận gộp.

        **Hàm ý kinh doanh**

        - Thay vì giảm toàn bộ ngân sách khuyến mãi, nên rà soát logic của nhóm khuyến mãi giảm tiền cố định trước tiên.
        - KPI cần theo dõi là `lợi nhuận gộp theo từng loại khuyến mãi`, không chỉ mức tăng doanh thu.
        """
    ),
    md_cell(
        """
        ## 4. Phân tích ma sát thương mại: huỷ đơn và trả hàng đang làm rò rỉ doanh thu ở đâu?

        Đây là phần chẩn đoán mạnh nhất cho bài toán kinh doanh: tìm đúng nơi doanh thu bị mất sau khi nhu cầu đã tới.
        """
    ),
    code_cell(
        """
        streetwear_refund_share = (
            returns_enriched.loc[returns_enriched["category"] == "Streetwear", "refund_amount"].sum()
            / returns_enriched["refund_amount"].sum()
            * 100
        )
        wrong_size_share = return_reason.get("wrong_size", 0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].bar(payment_cancel.index.astype(str), payment_cancel.values, color="#e76f51")
        axes[0].set_title("Tỷ lệ huỷ đơn theo phương thức thanh toán")
        axes[0].set_xlabel("Phương thức thanh toán")
        axes[0].set_ylabel("Tỷ lệ huỷ đơn (%)")
        axes[0].tick_params(axis="x", rotation=25)

        axes[1].barh(return_reason.head(5).sort_values().index.astype(str), return_reason.head(5).sort_values().values, color="#f4a261")
        axes[1].set_title("Nhóm lý do trả hàng lớn nhất")
        axes[1].set_xlabel("Tỷ trọng dòng trả hàng (%)")
        axes[1].set_ylabel("Lý do trả hàng")

        axes[2].bar(size_return_rate.index.astype(str), size_return_rate.values, color="#90be6d")
        axes[2].set_title("Tỷ lệ trả hàng theo size")
        axes[2].set_xlabel("Size")
        axes[2].set_ylabel("Tỷ lệ trả hàng (%)")

        plt.tight_layout()
        plt.show()

        print(f"Tỷ lệ huỷ COD = {payment_cancel.loc['cod']:.2f}% so với thẻ tín dụng = {payment_cancel.loc['credit_card']:.2f}%.")
        print(f"Tỷ trọng `wrong_size` trên toàn bộ lượt trả hàng = {wrong_size_share:.2f}%.")
        print(f"Tỷ trọng hoàn tiền của Streetwear trên toàn bộ khoản hoàn = {streetwear_refund_share:.2f}%.")
        print()
        print("Tỷ lệ trả hàng theo danh mục (%)")
        print(category_return_rate.round(2).to_string())
        """
    ),
    md_cell(
        """
        **Diễn giải**

        - COD là kênh thanh toán rủi ro cao nhất về huỷ đơn.
        - `wrong_size` là lý do lớn nhất của trả hàng, và Streetwear là nơi nỗi đau vận hành tập trung mạnh nhất.

        **Hàm ý kinh doanh**

        - Tạo khuyến khích cho trả trước hoặc giới hạn COD ở nhóm khách / khu vực rủi ro cao.
        - Ưu tiên bảng size, trợ lý chọn dáng, QC và nội dung trang chi tiết sản phẩm cho Streetwear trước, thay vì làm dàn trải cho toàn bộ danh mục.
        """
    ),
    md_cell(
        """
        ## 5. Bản đồ danh mục sản phẩm: nhóm nào vừa quan trọng vừa có rủi ro vận hành?

        Mục tiêu ở đây là gom nhiều tín hiệu vào một khung ra quyết định:

        - quy mô doanh thu,
        - biên lợi nhuận,
        - tỷ lệ trả hàng,
        - thiếu hàng / dư hàng.
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        bubble = axes[0].scatter(
            category_portfolio["return_rate"] * 100,
            category_portfolio["margin"] * 100,
            s=category_portfolio["rev_share"] * 6000,
            c=category_portfolio["days_of_supply"],
            cmap="viridis",
            alpha=0.8,
            edgecolor="black",
        )
        for cat, row in category_portfolio.iterrows():
            axes[0].annotate(cat, (row["return_rate"] * 100, row["margin"] * 100), fontsize=9, xytext=(5, 5), textcoords="offset points")
        axes[0].set_title("Bản đồ danh mục sản phẩm")
        axes[0].set_xlabel("Tỷ lệ trả hàng (%)")
        axes[0].set_ylabel("Biên lợi nhuận gộp (%)")
        cbar = plt.colorbar(bubble, ax=axes[0])
        cbar.set_label("Số ngày đủ hàng")

        risk_flags = category_portfolio[["stockout_flag", "overstock_flag"]].mul(100)
        risk_flags.plot(kind="bar", ax=axes[1], color=["#d62828", "#457b9d"])
        axes[1].set_title("Cờ rủi ro tồn kho theo danh mục")
        axes[1].set_xlabel("Danh mục")
        axes[1].set_ylabel("Tỷ lệ cờ trung bình (%)")
        axes[1].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        plt.show()

        print(
            (
                category_portfolio[
                    ["revenue", "rev_share", "margin", "return_rate", "stockout_flag", "overstock_flag", "days_of_supply"]
                ]
                .assign(
                    rev_share=lambda df: df["rev_share"] * 100,
                    margin=lambda df: df["margin"] * 100,
                    return_rate=lambda df: df["return_rate"] * 100,
                    stockout_flag=lambda df: df["stockout_flag"] * 100,
                    overstock_flag=lambda df: df["overstock_flag"] * 100,
                )
                .round(2)
                .to_string()
            )
        )
        """
    ),
    md_cell(
        """
        **Diễn giải**

        - Streetwear vừa là đầu tàu doanh thu lớn nhất, vừa là cụm rủi ro chính về trả hàng và tồn kho.
        - Việc cùng lúc có stockout và overstock cao cho thấy bài toán là **phân bổ và cơ cấu hàng hóa**, không phải chỉ là "thiếu hàng".

        **Hàm ý kinh doanh**

        - Streetwear phải là danh mục ưu tiên số 1 cho cả trưng bày hàng hóa, nội dung chọn dáng và phân bổ tồn kho.
        - Outdoor và các nhóm có số ngày đủ hàng quá cao nên được rà soát về quy tắc bổ sung hàng và chiến lược xả hàng.
        """
    ),
    md_cell(
        """
        ## 6. Lớp khuyến nghị: danh sách hành động đã được lượng hóa

        Đây là phần giúp nâng bài lên mức **khuyến nghị hành động** trong rubric: mỗi việc làm đều có lý do, KPI và mức độ ưu tiên tương đối.
        """
    ),
    code_cell(
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(action_df["initiative"], action_df["annual_value_vnd"], color=["#e76f51", "#2a9d8f", "#457b9d"])
        ax.set_title("Danh sách cơ hội đã được lượng hóa")
        ax.set_xlabel("Giá trị kinh tế ước tính (VND)")
        ax.set_ylabel("Hành động")
        ax.xaxis.set_major_formatter(BILLION_FMT)
        plt.tight_layout()
        plt.show()

        display_df = action_df.rename(
            columns={
                "initiative": "hành_động",
                "annual_value_vnd": "giá_trị_vnd",
                "kpi": "kpi",
                "why_now": "lý_do_ưu_tiên",
            }
        ).copy()
        display_df["giá_trị_vnd"] = display_df["giá_trị_vnd"].round(2)
        print(display_df.to_string(index=False))
        """
    ),
    md_cell(
        """
        ## Ghi chú cuối

        Cách dùng notebook này trong report:

        - Lấy `1-2` biểu đồ từ phần Nhu cầu / Chuyển đổi để chứng minh bạn có cả tư duy mô tả và tư duy dự báo.
        - Lấy `1-2` biểu đồ từ phần Khuyến mãi / Ma sát thương mại / Danh mục sản phẩm để thể hiện chiều sâu chẩn đoán và khuyến nghị.
        - Kết nối trực tiếp các phát hiện này với phần dự báo: mùa vụ, mức độ tập trung danh mục và ma sát vận hành giải thích vì sao mô hình cần kiểm định nghiêm ngặt và khả năng giải thích.
        """
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(ROOT / "eda_round1.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote eda_round1.ipynb")
