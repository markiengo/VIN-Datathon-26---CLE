import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
OUT = ROOT / "report" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


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

NOTEBOOK_COLORS = {
    "muted_gray": "#b8c0cc",
    "deep_teal": "#006d77",
    "seafoam": "#83c5be",
    "teal": "#2a9d8f",
    "sand": "#e9c46a",
    "orange": "#f4a261",
    "red": "#e76f51",
    "blue": "#457b9d",
}


def load_data():
    sales = pd.read_csv(DATA / "sales.csv", parse_dates=["Date"])
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
    order_items = pd.read_csv(DATA / "order_items.csv", low_memory=False)
    products = pd.read_csv(DATA / "products.csv")
    promotions = pd.read_csv(DATA / "promotions.csv", parse_dates=["start_date", "end_date"])
    returns = pd.read_csv(DATA / "returns.csv", parse_dates=["return_date"])
    customers = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
    geography = pd.read_csv(DATA / "geography.csv")
    inventory = pd.read_csv(DATA / "inventory.csv", parse_dates=["snapshot_date"])
    web = pd.read_csv(DATA / "web_traffic.csv", parse_dates=["date"])
    return sales, orders, order_items, products, promotions, returns, customers, geography, inventory, web


def prepare_tables():
    sales, orders, order_items, products, promotions, returns, customers, geography, inventory, web = load_data()

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

    # Channel quality stats for report text
    ch_quality = order_level.groupby("order_source").agg(
        aov=("order_revenue", "mean"),
        cancel_rate=("cancelled", "mean"),
        n=("order_id", "count"),
    )
    ch_quality = ch_quality[ch_quality["n"] >= 2000]
    ch_quality["cancel_pct"] = ch_quality["cancel_rate"] * 100

    rates = order_level.groupby("payment_method")["cancelled"].mean()
    cod_orders = order_level.loc[order_level["payment_method"] == "cod"]
    cod_revenue_at_risk = max(rates.get("cod", 0) - rates.get("credit_card", 0), 0) * len(cod_orders) * cod_orders.loc[
        cod_orders["cancelled"], "order_revenue"
    ].mean()
    streetwear_wrong_size_refund = returns_enriched.loc[
        (returns_enriched["category"] == "Streetwear") & (returns_enriched["return_reason"] == "wrong_size"),
        "refund_amount",
    ].sum()

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
        "aov_vnd": float(items.groupby("order_id")["revenue"].sum().mean()),
        "units_per_order": float(order_items["quantity"].sum() / order_items["order_id"].nunique()),
        "promo_penetration_pct": float(order_items["promo_id"].notna().mean() * 100),
        "repeat_purchase_rate_pct": float(
            orders.groupby("customer_id")["order_id"].count().gt(1).mean() * 100
        ),
        "return_rate_order_pct": float(
            returns["order_id"].nunique()
            / orders.loc[orders["order_status"] == "delivered", "order_id"].nunique()
            * 100
        ),
        "cancel_rate_pct": float(
            orders.loc[orders["order_status"] == "cancelled", "order_id"].nunique()
            / orders["order_id"].nunique()
            * 100
        ),
        "best_channel_aov_name": str(ch_quality["aov"].idxmax()),
        "best_channel_aov_vnd": float(ch_quality["aov"].max()),
        "best_channel_cancel_name": str(ch_quality["cancel_pct"].idxmin()),
        "best_channel_cancel_pct": float(ch_quality["cancel_pct"].min()),
        "worst_channel_cancel_name": str(ch_quality["cancel_pct"].idxmax()),
        "worst_channel_cancel_pct": float(ch_quality["cancel_pct"].max()),
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
        "metrics": metrics,
        "orders": orders,
        "customers": customers,
        "items": items,
        "order_level": order_level,
        "returns_enriched": returns_enriched,
    }


def build_margin_channel_figure(bundle):
    """Left: GP margin % by category. Right: AOV + cancel rate by top channels."""
    items = bundle["items"]
    orders = bundle["orders"]
    order_level = bundle["order_level"]
    returns_enriched = bundle["returns_enriched"]

    active = items[items["order_status"] != "cancelled"].copy()
    active["cogs_line"] = active["cogs"] * active["quantity"]
    cat = active.groupby("category").agg(
        revenue=("revenue", "sum"),
        cogs=("cogs_line", "sum"),
        discounts=("discount_amount", "sum"),
    ).reset_index()
    cat_refunds = returns_enriched.groupby("category")["refund_amount"].sum().rename("refunds").reset_index()
    cat = cat.merge(cat_refunds, on="category", how="left").fillna({"refunds": 0})
    cat["net_gp"] = cat["revenue"] - cat["cogs"] - cat["discounts"] - cat["refunds"]
    cat["margin_pct"] = cat["net_gp"] / cat["revenue"] * 100
    cat = cat.sort_values("revenue", ascending=True)

    order_src = orders[["order_id", "order_source"]].merge(
        order_level[["order_id", "cancelled"]], on="order_id"
    )
    ch_cancel = order_src.groupby("order_source")["cancelled"].mean().mul(100)
    ch_rev = items.groupby("order_source")["revenue"].sum()
    ch_cnt = orders.groupby("order_source")["order_id"].nunique()
    ch_aov = (ch_rev / ch_cnt).dropna()
    ch_df = pd.DataFrame({"aov": ch_aov, "cancel_rate": ch_cancel, "n_orders": ch_cnt}).dropna()
    ch_df = ch_df[ch_df["n_orders"] >= 2000].sort_values("aov", ascending=True)

    fig, axes = plt.subplots(2, 1, figsize=(6.25, 5.3), gridspec_kw={"height_ratios": [1.05, 0.95]})

    colors = [NOTEBOOK_COLORS["red"] if m < 0 else NOTEBOOK_COLORS["teal"] for m in cat["margin_pct"]]
    bars = axes[0].barh(cat["category"], cat["margin_pct"], color=colors)
    axes[0].axvline(0, color="black", lw=0.8)
    for bar, val in zip(bars, cat["margin_pct"]):
        xpos = val + 0.5 if val >= 0 else val - 0.5
        ha = "left" if val >= 0 else "right"
        axes[0].text(xpos, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", ha=ha, fontsize=8)
    axes[0].set_title("Biên lợi nhuận gộp thực theo danh mục\n(sau COGS, giảm giá và hoàn tiền)")
    axes[0].set_xlabel("Biên GP (%)")

    x = np.arange(len(ch_df))
    ax2 = axes[1].twiny()
    axes[1].barh(x, ch_df["aov"] / 1000, height=0.48, color=NOTEBOOK_COLORS["blue"], label="AOV (nghìn VND)")
    ax2.scatter(ch_df["cancel_rate"], x, color=NOTEBOOK_COLORS["red"], s=28, label="Tỷ lệ hủy (%)", zorder=3)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(ch_df.index, fontsize=9)
    axes[1].set_xlabel("AOV (nghìn VND)")
    ax2.set_xlabel("Tỷ lệ hủy (%)", labelpad=6)
    axes[1].set_title("Chất lượng kênh: AOV vs. tỷ lệ hủy đơn")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].grid(axis="x", alpha=0.18)
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right", frameon=False)

    plt.tight_layout()
    plt.savefig(OUT / "margin_channel.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def build_cohort_figure(orders):
    """New vs. repeat order split (annual stacked bar) + cohort retention heatmap."""
    first_order = orders.groupby("customer_id")["order_date"].min().rename("first_order_date")
    oc = orders.merge(first_order, on="customer_id")
    oc["months_since_first"] = (
        (oc["order_date"].dt.year - oc["first_order_date"].dt.year) * 12
        + oc["order_date"].dt.month - oc["first_order_date"].dt.month
    )
    oc["is_new"] = oc["months_since_first"] == 0
    oc["year"] = oc["order_date"].dt.year
    oc["cohort_year"] = oc["first_order_date"].dt.year

    # Annual new vs repeat order share
    annual = oc.groupby(["year", "is_new"])["order_id"].count().unstack(fill_value=0)
    annual.columns = ["repeat", "new"]
    annual["total"] = annual["new"] + annual["repeat"]
    annual["new_pct"] = annual["new"] / annual["total"]
    annual["repeat_pct"] = annual["repeat"] / annual["total"]

    # Cohort retention: % of cohort customers who ordered again at M+1, M+3, M+6, M+12
    cohort_size = (
        oc[oc["months_since_first"] == 0]
        .groupby("cohort_year")["customer_id"].nunique()
    )
    offsets = [1, 3, 6, 12]
    retention_dict = {}
    for m in offsets:
        active = (
            oc[oc["months_since_first"] == m]
            .groupby("cohort_year")["customer_id"].nunique()
        )
        retention_dict[f"M+{m}"] = (active / cohort_size).fillna(0)
    retention_pivot = (
        pd.DataFrame(retention_dict)
        .loc[2013:2021]
        .mul(100)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    years = annual.index.astype(str)
    axes[0].bar(years, annual["repeat_pct"] * 100, label="Khách quay lại", color=NOTEBOOK_COLORS["teal"])
    axes[0].bar(
        years, annual["new_pct"] * 100,
        bottom=annual["repeat_pct"] * 100,
        label="Khách mới", color=NOTEBOOK_COLORS["sand"],
    )
    axes[0].set_title("Tỷ trọng đơn hàng: khách mới vs. quay lại theo năm")
    axes[0].set_ylabel("Tỷ trọng (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False)
    axes[0].tick_params(axis="x", rotation=45)
    for i, (_, row) in enumerate(annual.iterrows()):
        if row["total"] > 0:
            axes[0].text(
                i, row["repeat_pct"] * 50, f"{row['repeat_pct']*100:.0f}%",
                ha="center", va="center", fontsize=7.5, color="white", fontweight="bold",
            )
            axes[0].text(
                i, row["repeat_pct"] * 100 + row["new_pct"] * 50, f"{row['new_pct']*100:.0f}%",
                ha="center", va="center", fontsize=7.5, color="black",
            )

    sns.heatmap(
        retention_pivot,
        ax=axes[1],
        annot=True,
        fmt=".1f",
        cmap="YlOrRd_r",
        linewidths=0.4,
        cbar_kws={"label": "% giữ chân"},
        vmin=0,
        vmax=retention_pivot.max().max(),
    )
    axes[1].set_title("Retention theo cohort năm (% khách đặt lại ở M+1/3/6/12)")
    axes[1].set_xlabel("Thời điểm sau lần mua đầu")
    axes[1].set_ylabel("Năm cohort")

    plt.tight_layout()
    plt.savefig(OUT / "cohort_retention.png", bbox_inches="tight")
    plt.close(fig)


def save_figures(bundle):  # noqa: C901
    month_labels = bundle["month_labels"]
    weekday_vi = bundle["weekday_vi"]
    annual_avg = bundle["annual_avg"]
    monthly_recent = bundle["monthly_recent"]
    dow_recent = bundle["dow_recent"]
    promo_summary = bundle["promo_summary"]
    promo_type = bundle["promo_type"]
    promo_channel = bundle["promo_channel"]
    payment_cancel = bundle["payment_cancel"]
    return_reason = bundle["return_reason"]
    size_return_rate = bundle["size_return_rate"]
    category_portfolio = bundle["category_portfolio"]
    orders = bundle["orders"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    year_colors = [NOTEBOOK_COLORS["muted_gray"] if y < 2019 else NOTEBOOK_COLORS["deep_teal"] for y in annual_avg["year"]]
    axes[0].bar(annual_avg["year"].astype(str), annual_avg["Revenue"], color=year_colors)
    axes[0].set_title("Doanh thu ngày trung bình theo năm")
    axes[0].set_ylabel("Doanh thu")
    axes[0].yaxis.set_major_formatter(BILLION_FMT)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].plot(month_labels, monthly_recent["Revenue"].values, marker="o", color=NOTEBOOK_COLORS["deep_teal"], lw=2)
    axes[1].set_title("Doanh thu ngày trung bình theo tháng (2019-2022)")
    axes[1].set_ylabel("Doanh thu")
    axes[1].yaxis.set_major_formatter(MILLION_FMT)

    axes[2].bar([weekday_vi[d] for d in dow_recent.index.astype(str)], dow_recent["Revenue"].values, color=NOTEBOOK_COLORS["seafoam"])
    axes[2].set_title("Doanh thu ngày trung bình theo ngày trong tuần (2019-2022)")
    axes[2].set_ylabel("Doanh thu")
    axes[2].yaxis.set_major_formatter(MILLION_FMT)
    axes[2].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUT / "demand_baseline.png", bbox_inches="tight")
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

    # Panel 3: net GP margin by category (Section 7)
    _it = bundle["items"]
    _re = bundle["returns_enriched"]
    _act = _it[_it["order_status"] != "cancelled"].copy()
    _act["cogs_line"] = _act["cogs"] * _act["quantity"]
    _cat = _act.groupby("category").agg(
        revenue=("revenue", "sum"),
        cogs=("cogs_line", "sum"),
        discounts=("discount_amount", "sum"),
    ).reset_index()
    _cat_ref = _re.groupby("category")["refund_amount"].sum().rename("refunds").reset_index()
    _cat = _cat.merge(_cat_ref, on="category", how="left").fillna({"refunds": 0})
    _cat["margin_pct"] = (_cat["revenue"] - _cat["cogs"] - _cat["discounts"] - _cat["refunds"]) / _cat["revenue"] * 100
    _cat = _cat.sort_values("revenue", ascending=True)
    _gc = ["#e76f51" if m < 0 else "#2a9d8f" for m in _cat["margin_pct"]]
    axes[2].barh(_cat["category"], _cat["margin_pct"], color=_gc)
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set_title("Biên GP thực theo danh mục\n(sau COGS, giảm giá, hoàn tiền)")
    axes[2].set_xlabel("Biên GP (%)")
    for i, (_, row) in enumerate(_cat.iterrows()):
        xpos = row["margin_pct"] + 0.3 if row["margin_pct"] >= 0 else row["margin_pct"] - 0.3
        axes[2].text(xpos, i, f"{row['margin_pct']:.1f}%", va="center",
                     ha="left" if row["margin_pct"] >= 0 else "right", fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "returns_cancellations.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.15, 6.2), height_ratios=[1.1, 0.82])
    bubble = axes[0].scatter(
        category_portfolio["return_rate"] * 100,
        category_portfolio["margin"] * 100,
        s=category_portfolio["rev_share"] * 4700,
        c=category_portfolio["days_of_supply"],
        cmap="viridis",
        alpha=0.82,
        edgecolor="black",
    )
    for cat, row in category_portfolio.iterrows():
        axes[0].annotate(f"{cat}\nSO {row['stockout_flag'] * 100:.0f}% | OS {row['overstock_flag'] * 100:.0f}%", (row["return_rate"] * 100, row["margin"] * 100), fontsize=7.0, xytext=(4, 4), textcoords="offset points")
    axes[0].set_title("Bản đồ danh mục & tồn kho")
    axes[0].set_xlabel("Tỷ lệ trả hàng (%)")
    axes[0].set_ylabel("Biên lợi nhuận gộp (%)")
    cbar = plt.colorbar(bubble, ax=axes[0])
    cbar.set_label("Số ngày đủ hàng")

    risk_flags = category_portfolio[["stockout_flag", "overstock_flag"]].mul(100)
    risk_flags.plot(kind="bar", ax=axes[1], color=["#d62828", "#457b9d"])
    axes[1].set_title("Cờ stockout / overstock theo danh mục")
    axes[1].set_ylabel("Tỷ lệ cờ trung bình (%)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_xlabel("")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "category_portfolio.png", bbox_inches="tight")
    plt.close(fig)

    build_cohort_figure(orders)
    build_margin_channel_figure(bundle)

def main():
    bundle = prepare_tables()
    save_figures(bundle)
    (OUT / "summary_metrics.json").write_text(json.dumps(bundle["metrics"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Assets saved to {OUT}")


if __name__ == "__main__":
    main()
