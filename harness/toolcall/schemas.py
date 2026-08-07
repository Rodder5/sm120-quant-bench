"""Tool battery for the tool-calling-under-quantization study.

Twelve synthetic tools in OpenAI function-calling format, graded by difficulty.
The near-miss pair (schedule_meeting / schedule_reminder) exists to probe
selection confusion; transfer_funds exists to tie into the repo's existing
finding that 4-bit damage pools in near-tie digit choices.

Everything here is static data. No randomness, no timestamps.
"""

TOOLS = [
    # -- trivial: single required string arg ---------------------------------
    {
        "type": "function",
        "function": {
            "name": "define_word",
            "description": "Look up the dictionary definition of a single English word.",
            "parameters": {
                "type": "object",
                "properties": {"word": {"type": "string", "description": "The word to define."}},
                "required": ["word"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name."}},
                "required": ["city"],
            },
        },
    },
    # -- multi-arg, mixed types, required vs optional ------------------------
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight between two cities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Departure city."},
                    "destination": {"type": "string", "description": "Arrival city."},
                    "passengers": {"type": "integer", "description": "Number of passengers."},
                    "refundable": {"type": "boolean", "description": "Whether the fare must be refundable."},
                    "max_price": {"type": "number", "description": "Maximum ticket price in dollars."},
                },
                "required": ["origin", "destination", "passengers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_invoice",
            "description": "Create an invoice for a client.",
            "parameters": {
                "type": "object",
                "properties": {
                    "client": {"type": "string", "description": "Client name."},
                    "amount": {"type": "number", "description": "Invoice amount in dollars."},
                    "taxable": {"type": "boolean", "description": "Whether tax applies."},
                    "due_days": {"type": "integer", "description": "Days until the invoice is due."},
                },
                "required": ["client", "amount", "taxable"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resize_image",
            "description": "Resize an image file to the given dimensions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the image file."},
                    "width": {"type": "integer", "description": "Target width in pixels."},
                    "height": {"type": "integer", "description": "Target height in pixels."},
                    "keep_aspect": {"type": "boolean", "description": "Preserve aspect ratio by padding."},
                    "quality": {"type": "number", "description": "Output quality from 0.0 to 1.0."},
                },
                "required": ["path", "width", "height"],
            },
        },
    },
    # -- enums ----------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "set_thermostat",
            "description": "Set the home thermostat mode and optionally a target temperature.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["heat", "cool", "auto", "off"],
                             "description": "Operating mode."},
                    "temperature": {"type": "number", "description": "Target temperature in celsius."},
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text into a target language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to translate."},
                    "target_language": {
                        "type": "string",
                        "enum": ["french", "german", "spanish", "italian", "portuguese",
                                 "dutch", "japanese", "korean", "mandarin", "hindi",
                                 "arabic", "swahili"],
                        "description": "Language to translate into.",
                    },
                },
                "required": ["text", "target_language"],
            },
        },
    },
    # -- nested objects and arrays -------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "Create a purchase order for a customer with one or more line items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Customer full name."},
                            "email": {"type": "string", "description": "Customer email address."},
                        },
                        "required": ["name", "email"],
                    },
                    "items": {
                        "type": "array",
                        "description": "Line items.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string", "description": "Product code."},
                                "qty": {"type": "integer", "description": "Quantity ordered."},
                            },
                            "required": ["sku", "qty"],
                        },
                    },
                    "notes": {"type": "string", "description": "Free-text order notes."},
                },
                "required": ["customer", "items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "configure_alert",
            "description": "Configure a monitoring alert with warning and critical thresholds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "description": "Metric name to monitor."},
                    "thresholds": {
                        "type": "object",
                        "properties": {
                            "warning": {"type": "number", "description": "Warning threshold."},
                            "critical": {"type": "number", "description": "Critical threshold."},
                        },
                        "required": ["warning", "critical"],
                    },
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Notification channels, e.g. email addresses or #channels.",
                    },
                },
                "required": ["metric", "thresholds", "channels"],
            },
        },
    },
    # -- near-miss pair: confusable names, overlapping schemas ---------------
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Schedule a meeting with other people at a specific time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Meeting title."},
                    "start_time": {"type": "string", "description": "Start time, format YYYY-MM-DD HH:MM."},
                    "duration_minutes": {"type": "integer", "description": "Length of the meeting in minutes."},
                    "attendees": {"type": "array", "items": {"type": "string"},
                                  "description": "Names of the other attendees."},
                },
                "required": ["title", "start_time", "duration_minutes", "attendees"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_reminder",
            "description": "Schedule a personal reminder for yourself at a specific time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Reminder text."},
                    "remind_at": {"type": "string", "description": "When to fire, format YYYY-MM-DD HH:MM."},
                    "repeat_daily": {"type": "boolean", "description": "Repeat every day at the same time."},
                },
                "required": ["title", "remind_at"],
            },
        },
    },
    # -- numeric precision ----------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "transfer_funds",
            "description": "Transfer money to an account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to transfer in dollars."},
                    "account_id": {"type": "string",
                                   "description": "Destination account number, a string of digits."},
                    "memo": {"type": "string", "description": "Optional transfer memo."},
                },
                "required": ["amount", "account_id"],
            },
        },
    },
]

TOOLS_BY_NAME = {t["function"]["name"]: t for t in TOOLS}
TOOL_NAMES = list(TOOLS_BY_NAME)

# The languages of translate_text's enum, importable by the gold generator so
# message text and enum values can never drift apart.
TRANSLATE_LANGUAGES = TOOLS_BY_NAME["translate_text"]["function"]["parameters"][
    "properties"]["target_language"]["enum"]
THERMOSTAT_MODES = TOOLS_BY_NAME["set_thermostat"]["function"]["parameters"][
    "properties"]["mode"]["enum"]
