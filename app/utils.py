import json

def json_parse(jsonstr: str):
    try:
        return json.loads(jsonstr)
    except json.JSONDecodeError:
        return {
            "error": "JSON 파싱 실패. 응답 형식이 올바르지 않음.",
            "raw": jsonstr
        }
