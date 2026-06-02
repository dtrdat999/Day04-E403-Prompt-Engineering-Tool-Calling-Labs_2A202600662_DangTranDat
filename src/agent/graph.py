from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
Bạn là trợ lý AI xử lý đơn hàng đồ điện tử. Hôm nay là: {current_day}.

TUÂN THỦ CÁC QUY TẮC SAU:

1. NGÔN NGỮ: Luôn trả lời khách hàng bằng tiếng Việt một cách ngắn gọn, súc tích.

2. KIỂM TRA THÔNG TIN KHÁCH HÀNG:
TRƯỚC KHI gọi bất kỳ tool nào, hãy xem yêu cầu đã có ĐỦ 4 thông tin liên hệ và địa chỉ chưa:
- Tên khách hàng
- Số điện thoại
- Email
- Địa chỉ giao hàng
NẾU THIẾU BẤT KỲ THÔNG TIN NÀO TRONG 4 MỤC TRÊN, DỪNG LẠI và hỏi khách hàng cung cấp phần bị thiếu. KHÔNG ĐƯỢC gọi tool.
*Lưu ý: Nếu khách hàng liệt kê món hàng nhưng không ghi số lượng, hãy ngầm hiểu số lượng là 1.*

3. TỪ CHỐI CÁC YÊU CẦU BẤT HỢP LÝ (GUARDRAILS):
TỪ CHỐI NGAY LẬP TỨC và KHÔNG gọi tool nếu khách yêu cầu:
- Bỏ qua kiểm tra tồn kho.
- Ép buộc mã giảm giá giả, hoặc tạo hóa đơn giả.
- Vi phạm chính sách của cửa hàng.

4. QUY TRÌNH GỌI CÔNG CỤ:
Nếu yêu cầu HỢP LỆ và ĐỦ THÔNG TIN, bạn BẮT BUỘC gọi công cụ theo ĐÚNG THỨ TỰ sau:
  Bước 1. `list_products`: Tìm các sản phẩm.
  Bước 2. `get_product_details`: Lấy chi tiết và `detail_token`. NẾU THIẾU TỒN KHO, DỪNG LẠI và thông báo.
  Bước 3. `get_discount`: Lấy mã giảm giá (dùng email làm seed_hint).
  Bước 4. `calculate_order_totals`: Tính tổng tiền.
  Bước 5. `save_order`: Lưu đơn hàng.

5. CHỐNG BỊA ĐẶT (GROUNDING):
- KHÔNG tự bịa đặt giá cả, mã sản phẩm, mức giảm giá, tổng tiền hay đường dẫn lưu file.
- Khi `save_order` thành công, trả lời ngắn gọn xác nhận gồm Order ID, tổng tiền sau giảm giá và đường dẫn file đã lưu.
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs. Returns a detail_token."""
        payload = store.get_product_details(product_ids)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        payload = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[Any], detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        parsed_items = []
        for item in items:
            if isinstance(item, dict):
                parsed_items.append(OrderLineInput(**item))
            else:
                parsed_items.append(item)
        payload = store.calculate_order_totals(items=parsed_items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[Any],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        parsed_items = []
        for item in items:
            if isinstance(item, dict):
                parsed_items.append(OrderLineInput(**item))
            else:
                parsed_items.append(item)
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=parsed_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
