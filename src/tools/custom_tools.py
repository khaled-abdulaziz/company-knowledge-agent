# ==============================================================
# Custom Python Action Tools
# ==============================================================


# ==============================================================
# Tool: format_employee_table
# ==============================================================

def format_employee_table(rows: list[dict]) -> str:
    """
    Formats a list of employee dicts into a readable text table.

    Args:
        rows (list[dict]): Employee records from MySQL.

    Returns:
        str: Formatted table string, or a "no data" message.

    Usage:
        rows = [{"name": "Ahmed", "department": "HR", "position": "Manager"}]
        print(format_employee_table(rows))
    """
    if not rows:    #if rows is empty
        return "⚠️ No records found."

    headers = list(rows[0].keys())
    col_widths = {  
        h: max(len(str(h)), max(len(str(row.get(h, ""))) for row in rows))
        for h in headers
    }

    separator = "+" + "+".join("-" * (col_widths[h] + 2) for h in headers) + "+"  
    header_row = "|" + "|".join(f" {h:<{col_widths[h]}} " for h in headers) + "|"

    lines = [separator, header_row, separator]
    for row in rows:
        data_row = "|" + "|".join(
            f" {str(row.get(h, '')):<{col_widths[h]}} " for h in headers
        ) + "|"
        lines.append(data_row)
    lines.append(separator)

    return "\n".join(lines)


# ==============================================================
# Tool: summarize_results
# ==============================================================

def summarize_results(rows: list[dict], key_field: str = "name") -> str: 
    """
    Collapses a list of dicts into a short comma-separated summary.
    Useful for generating a quick overview before a detailed answer.

    Args:
        rows      (list[dict]): Records to summarize.
        key_field (str)       : Which field to show in the summary (default "name").

    Returns:
        str: "Found 3 records: Ahmed, Sara, Mohammed" style string.

    Usage:
        summarize_results(rows, key_field="name")
        → "Found 3 records: Ahmed, Sara, Mohammed"
    """
    if not rows:
        return "No records found."

    count  = len(rows)
    values = [str(row.get(key_field, "?")) for row in rows[:5]] # take only first 5 rows and Loops through first 5 rows. .get() returns default "?"
    suffix = f" (and {count - 5} more)" if count > 5 else ""   # IF more than 5 records: show extra count ELSE:show nothing

    return f"Found {count} record{'s' if count != 1 else ''}: {', '.join(values)}{suffix}" # This line builds final English sentence.


# ==============================================================
# Tool: detect_language
# ==============================================================

def detect_language(text: str) -> str:
    """
    Detects whether a string is Arabic or English.
    Simple heuristic — checks Unicode ranges for Arabic characters.

    Args:
        text (str): The user's question.

    Returns:
        str: "ar" for Arabic, "en" for English (default).

    Usage:
        detect_language("ما هي سياسة الإجازة؟")  → "ar"
        detect_language("What is the leave policy?") → "en"
    """
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF") 