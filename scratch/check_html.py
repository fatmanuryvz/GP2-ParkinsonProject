import html.parser

class HTMLTagChecker(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        # Ignore self-closing tags in HTML5
        if tag in ['img', 'br', 'hr', 'input', 'meta', 'link', 'source', 'col']:
            return
        self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag):
        if tag in ['img', 'br', 'hr', 'input', 'meta', 'link', 'source', 'col']:
            return
        if not self.stack:
            self.errors.append(f"Unexpected closing tag </{tag}> at line {self.getpos()[0]}")
            return
        last_tag, pos = self.stack.pop()
        if last_tag != tag:
            self.errors.append(f"Mismatched tag: opened <{last_tag}> at line {pos[0]}, closed with </{tag}> at line {self.getpos()[0]}")

    def check(self, content):
        self.feed(content)
        while self.stack:
            tag, pos = self.stack.pop()
            self.errors.append(f"Unclosed tag <{tag}> opened at line {pos[0]}")
        return self.errors

with open("parkinson_detection_app.html", "r", encoding="utf-8") as f:
    content = f.read()

checker = HTMLTagChecker()
errors = checker.check(content)
if errors:
    print("Found HTML balance errors:")
    for err in errors[:20]:
        print(err)
else:
    print("HTML is perfectly balanced!")
