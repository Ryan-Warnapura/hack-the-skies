import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from forge.__main__ import server_state


def test_existing_realityforge_and_other_app_are_distinguished():
    class Handler(BaseHTTPRequestHandler):
        body = b'{"name":"RealityForge","status":"ok"}'

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(self.body)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server_state(port) == 'running'
        Handler.body = b'<html>Another application</html>'
        assert server_state(port) == 'occupied'
        Handler.body = b'{"name":"Other app","status":"ok"}'
        assert server_state(port) == 'occupied'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_free_port_is_available():
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    assert server_state(port) == 'available'
