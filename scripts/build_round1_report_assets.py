import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent   # project root (scripts/ is one level below)
DATA = ROOT / "dataset"
OUT  = ROOT / "report_assets"
OUT.mkdir(exist_ok=True)


def fmt_billion(x, _):
    return f"{x / 1e9:.1f}B"


def fmt_million(x, _):
    return f"{x / 1e6:.1f}M"


BILLION_FMT = FuncFormatter(fmt_billion)
MILLION_FMT = FuncFormatter(fmt_million)

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    }
)


def load_data():
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"])
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
    order_items = pd.read_csv(DATA / "order_items.csv", low_memory=False)
    products = pd.read_csv(DATA / "products.csv")
    promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
    returns = pd.read_csv(DATA / "returns.csv", parse_dates=["return_date"])
    geography = pd.read_csv(DATA / "geography.csv")
    inventory = pd.read_csv(DATA / "inventory.csv", parse_dates=["snapshot_date"])
    web = pd.read_csv(DATA / "web_traffic.csv", parse_dates=["date"])
    return sales, orders, order_items, products, promotions, returns, geography, inventory, web


def prepare_tables():
    sales, orders, order_items, products, promotions, returns, geography, inventory, web = load_data()

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
    returned = returns.groupby("order_id").agg(return_lines=("return_id", "count"), refund_amount=("refund_amount", "sum"))
    order_level = orders.merge(order_revenue, left_on="order_id", right_index=True, how="left").merge(
        returned, left_on="order_id", right_index=True, how="left"
    )
    order_level[["order_revenue", "return_lines", "refund_amount"]] = order_level[
        ["order_revenue", "return_lines", "refund_amount"]
    ].fillna(0)
    order_level["cancelled"] = order_level["order_status"].eq("cancelled")
    order_level["returned"] = order_level["return_lines"].gt(0)

    daily_web = web.groupby("date").agg(
        sessions=("sessions", "sum"),
        visitors=("unique_visitors", "sum"),
        page_views=("page_views", "sum"),
    ).reset_index()
    daily_orders = orders.groupby("order_date").size().rename("orders").reset_index()
    daily = sales.merge(daily_web, left_on="Date", right_on="date", how="left").merge(
        daily_orders, left_on="Date", right_on="order_date", how="left"
    )
    daily["orders"] = daily["orders"].fillna(0)
    daily["conversion_proxy"] = daily["orders"] / daily["visitors"]
    daily["aov"] = daily["Revenue"] / daily["orders"].replace(0, np.nan)

    recent = sales.loc[sales["Date"] >= "2019-01-01"].copy()
    recent["year"] = recent["Date"].dt.year
    recent["month"] = recent["Date"].dt.month
    recent["dow"] = pd.Categorical(recent["Date"].dt.day_name(), categories=weekday_order, ordered=True)

    annual_avg = sales.assign(year=sales["Date"].dt.year).groupby("year")[["Revenue", "COGS"]].mean().reset_index()
    monthly_recent = recent.groupby("month")[["Revenue", "COGS"]].mean().reindex(range(1, 13))
    dow_recent = recent.groupby("dow")[["Revenue", "COGS"]].mean().reindex(weekday_order)

    returns_enriched = returns.merge(products[["product_id", "category", "segment", "size"]], on="product_id", how="left")
    return_reason = returns_enriched["return_reason"].value_counts(normalize=True).mul(100)
    size_return_rate = (returns_enriched.groupby("size").size() / items.groupby("size").size()).reindex(
        ["S", "M", "L", "XL"]
    ).mul(100)
    category_return_rate = (returns_enriched.groupby("category").size() / items.groupby("category").size()).sort_values(
        ascending=False
    ).mul(100)

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
    )
    category_portfolio["rev_share"] = category_portfolio["revenue"] / category_portfolio["revenue"].sum()
    category_portfolio["margin"] = category_portfolio["gross_profit"] / category_portfolio["revenue"]
    category_portfolio = category_portfolio.join(
        (returns_enriched.groupby("category").size() / items.groupby("category").size()).rename("return_rate")
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
                "initiative": "Giảm 10% hoàn tiền do sai size của Streetwear",
                "annual_value_vnd": streetwear_wrong_size_refund * 0.10,
                "kpi": "Giá trị hoàn tiền do sai size",
            },
            {
                "initiative": "Kéo tỷ lệ huỷ COD về gần mức thẻ",
                "annual_value_vnd": cod_revenue_at_risk,
                "kpi": "Tỷ lệ huỷ COD",
            },
            {
                "initiative": "Thiết kế lại hoặc dừng khuyến mãi giảm tiền cố định",
                "annual_value_vnd": fixed_gp_drag,
                "kpi": "Biên lợi nhuận gộp theo khuyến mãi",
            },
        ]
    ).sort_values("annual_value_vnd", ascending=True)

    metrics = {
        "pre_2019_avg_daily_revenue": float(sales.loc[sales["Date"] <= "2018-12-31", "Revenue"].mean()),
        "post_2019_avg_daily_revenue": float(sales.loc[sales["Date"] >= "2019-01-01", "Revenue"].mean()),
        "revenue_drop_pct_after_2018": float(
            (1 - sales.loc[sales["Date"] >= "2019-01-01", "Revenue"].mean()
             / sales.loc[sales["Date"] <= "2018-12-31", "Revenue"].mean()) * 100
        ),
        "corr_revenue_orders": float(daily[["Revenue", "orders"]].corr().iloc[0, 1]),
        "corr_revenue_sessions": float(daily[["Revenue", "sessions"]].corr().iloc[0, 1]),
        "corr_revenue_conversion_proxy": float(daily[["Revenue", "conversion_proxy"]].corr().iloc[0, 1]),
        "fixed_promo_margin_pct": float(promo_type.loc["fixed", "margin"] * 100),
        "fixed_promo_gp_vnd": float(promo_type.loc["fixed", "gross_profit"]),
        "cod_cancel_pct": float(payment_cancel.loc["cod"]),
        "credit_card_cancel_pct": float(payment_cancel.loc["credit_card"]),
        "streetwear_revenue_share_pct": float(category_portfolio.loc["Streetwear", "rev_share"] * 100),
        "streetwear_refund_share_pct": float(
            returns_enriched.loc[returns_enriched["category"] == "Streetwear", "refund_amount"].sum()
            / returns_enriched["refund_amount"].sum()
            * 100
        ),
        "streetwear_stockout_pct": float(category_portfolio.loc["Streetwear", "stockout_flag"] * 100),
        "streetwear_overstock_pct": float(category_portfolio.loc["Streetwear", "overstock_flag"] * 100),
        "streetwear_days_of_supply": float(category_portfolio.loc["Streetwear", "days_of_supply"]),
        "cod_revenue_at_risk_vnd": float(cod_revenue_at_risk),
        "streetwear_wrong_size_10pct_savings_vnd": float(streetwear_wrong_size_refund * 0.10),
        "weekday_best": str(dow_recent["Revenue"].idxmax()),
        "weekday_worst": str(dow_recent["Revenue"].idxmin()),
        "peak_month": int(monthly_recent["Revenue"].idxmax()),
        "trough_month": int(monthly_recent["Revenue"].idxmin()),
    }

    return {
        "month_labels": month_labels,
        "weekday_vi": weekday_vi,
        "annual_avg": annual_avg,
        "monthly_recent": monthly_recent,
        "dow_recent": dow_recent,
        "daily": daily,
        "promo_summary": promo_summary,
        "promo_type": promo_type,
        "promo_channel": promo_channel,
        "payment_cancel": payment_cancel,
        "return_reason": return_reason,
        "size_return_rate": size_return_rate,
        "category_return_rate": category_return_rate,
        "category_portfolio": category_portfolio,
        "action_df": action_df,
        "metrics": metrics,
    }


def save_figures(bundle):
    month_labels = bundle["month_labels"]
    weekday_vi = bundle["weekday_vi"]
    annual_avg = bundle["annual_avg"]
    monthly_recent = bundle["monthly_recent"]
    dow_recent = bundle["dow_recent"]
    daily = bundle["daily"]
    promo_summary = bundle["promo_summary"]
    promo_type = bundle["promo_type"]
    promo_channel = bundle["promo_channel"]
    payment_cancel = bundle["payment_cancel"]
    return_reason = bundle["return_reason"]
    size_return_rate = bundle["size_return_rate"]
    category_portfolio = bundle["category_portfolio"]
    action_df = bundle["action_df"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    year_colors = ["#c8d1dc" if y < 2019 else "#0b7285" for y in annual_avg["year"]]
    axes[0].bar(annual_avg["year"].astype(str), annual_avg["Revenue"], color=year_colors)
    axes[0].set_title("Doanh thu ngày trung bình theo năm")
    axes[0].set_ylabel("Doanh thu")
    axes[0].yaxis.set_major_formatter(BILLION_FMT)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].plot(month_labels, monthly_recent["Revenue"].values, marker="o", color="#0b7285", lw=2)
    axes[1].set_title("Doanh thu ngày trung bình theo tháng (2019-2022)")
    axes[1].set_ylabel("Doanh thu")
    axes[1].yaxis.set_major_formatter(MILLION_FMT)

    axes[2].bar([weekday_vi[d] for d in dow_recent.index.astype(str)], dow_recent["Revenue"].values, color="#74c0fc")
    axes[2].set_title("Doanh thu ngày trung bình theo ngày trong tuần (2019-2022)")
    axes[2].set_ylabel("Doanh thu")
    axes[2].yaxis.set_major_formatter(MILLION_FMT)
    axes[2].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUT / "demand_baseline.png", bbox_inches="tight")
    plt.close(fig)

    monthly_flow = daily.set_index("Date").resample("ME").sum(numeric_only=True)[["Revenue", "orders", "sessions"]]
    indexed_flow = monthly_flow / monthly_flow.iloc[0] * 100
    corr_orders = daily[["Revenue", "orders"]].corr().iloc[0, 1]
    corr_sessions = daily[["Revenue", "sessions"]].corr().iloc[0, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(indexed_flow.index, indexed_flow["Revenue"], label="Doanh thu", lw=2, color="#264653")
    axes[0].plot(indexed_flow.index, indexed_flow["orders"], label="Số đơn", lw=2, color="#2a9d8f")
    axes[0].plot(indexed_flow.index, indexed_flow["sessions"], label="Phiên truy cập", lw=2, color="#e9c46a")
    axes[0].set_title("Diễn biến chỉ số theo tháng")
    axes[0].set_ylabel("Chỉ số")
    axes[0].legend(frameon=False)

    axes[1].scatter(daily["orders"], daily["Revenue"], alpha=0.22, s=12, color="#2a9d8f")
    axes[1].set_title(f"Doanh thu và số đơn (corr={corr_orders:.2f})")
    axes[1].set_xlabel("Số đơn mỗi ngày")
    axes[1].set_ylabel("Doanh thu")
    axes[1].yaxis.set_major_formatter(BILLION_FMT)

    axes[2].scatter(daily["sessions"], daily["Revenue"], alpha=0.22, s=12, color="#f4a261")
    axes[2].set_title(f"Doanh thu và phiên truy cập (corr={corr_sessions:.2f})")
    axes[2].set_xlabel("Phiên truy cập mỗi ngày")
    axes[2].set_ylabel("Doanh thu")
    axes[2].yaxis.set_major_formatter(BILLION_FMT)
    plt.tight_layout()
    plt.savefig(OUT / "traffic_vs_orders.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(promo_summary.index, promo_summary["revenue"], color=["#adb5bd", "#457b9d"])
    axes[0].set_title("Doanh thu theo trạng thái dùng khuyến mãi")
    axes[0].set_ylabel("Doanh thu")
    axes[0].yaxis.set_major_formatter(BILLION_FMT)

    promo_margin_pct = promo_type["margin"].mul(100).sort_values(ascending=False)
    axes[1].bar(promo_margin_pct.index.astype(str), promo_margin_pct.values, color=["#2a9d8f", "#e76f51"])
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_title("Biên lợi nhuận gộp theo loại khuyến mãi")
    axes[1].set_ylabel("Biên (%)")
    for idx, value in enumerate(promo_margin_pct.values):
        axes[1].text(idx, value + (1 if value >= 0 else -4), f"{value:.1f}%", ha="center")

    top_channels = promo_channel.head(5).sort_values("revenue", ascending=True)
    axes[2].barh(top_channels.index.astype(str), top_channels["revenue"], color="#8ecae6")
    axes[2].set_title("Top kênh khuyến mãi theo doanh thu")
    axes[2].set_xlabel("Doanh thu")
    axes[2].xaxis.set_major_formatter(BILLION_FMT)
    plt.tight_layout()
    plt.savefig(OUT / "promo_economics.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(payment_cancel.index.astype(str), payment_cancel.values, color="#e76f51")
    axes[0].set_title("Tỷ lệ huỷ đơn theo phương thức thanh toán")
    axes[0].set_ylabel("Tỷ lệ huỷ đơn (%)")
    axes[0].tick_params(axis="x", rotation=25)

    top_reasons = return_reason.head(5).sort_values()
    axes[1].barh(top_reasons.index.astype(str), top_reasons.values, color="#f4a261")
    axes[1].set_title("Nhóm lý do trả hàng lớn nhất")
    axes[1].set_xlabel("Tỷ trọng dòng trả hàng (%)")

    axes[2].bar(size_return_rate.index.astype(str), size_return_rate.values, color="#90be6d")
    axes[2].set_title("Tỷ lệ trả hàng theo size")
    axes[2].set_ylabel("Tỷ lệ trả hàng (%)")
    plt.tight_layout()
    plt.savefig(OUT / "returns_cancellations.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bubble = axes[0].scatter(
        category_portfolio["return_rate"] * 100,
        category_portfolio["margin"] * 100,
        s=category_portfolio["rev_share"] * 7000,
        c=category_portfolio["days_of_supply"],
        cmap="viridis",
        alpha=0.82,
        edgecolor="black",
    )
    for cat, row in category_portfolio.iterrows():
        axes[0].annotate(cat, (row["return_rate"] * 100, row["margin"] * 100), fontsize=8, xytext=(5, 5), textcoords="offset points")
    axes[0].set_title("Bản đồ danh mục sản phẩm")
    axes[0].set_xlabel("Tỷ lệ trả hàng (%)")
    axes[0].set_ylabel("Biên lợi nhuận gộp (%)")
    cbar = plt.colorbar(bubble, ax=axes[0])
    cbar.set_label("Số ngày đủ hàng")

    risk_flags = category_portfolio[["stockout_flag", "overstock_flag"]].mul(100)
    risk_flags.plot(kind="bar", ax=axes[1], color=["#d62828", "#457b9d"])
    axes[1].set_title("Cờ rủi ro tồn kho theo danh mục")
    axes[1].set_ylabel("Tỷ lệ cờ trung bình (%)")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(OUT / "category_portfolio.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(action_df["initiative"], action_df["annual_value_vnd"], color=["#457b9d", "#2a9d8f", "#e76f51"])
    ax.set_title("Danh sách cơ hội đã được lượng hóa")
    ax.set_xlabel("Giá trị kinh tế ước tính (VND)")
    ax.xaxis.set_major_formatter(BILLION_FMT)
    plt.tight_layout()
    plt.savefig(OUT / "action_backlog.png", bbox_inches="tight")
    plt.close(fig)


def main():
    bundle = prepare_tables()
    save_figures(bundle)
    (OUT / "summary_metrics.json").write_text(json.dumps(bundle["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu biểu đồ và chỉ số vào {OUT}")


if __name__ == "__main__":
    main()
