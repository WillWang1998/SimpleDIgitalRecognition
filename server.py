import tornado.ioloop
import tornado.options
import tornado.httpserver
import tornado.autoreload
import tornado.web
import tornado.websocket
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pylab
import json
from importlib import reload
from tornado.options import define, options
from recognizer import Recognizer

reload(sys)
recognizer = Recognizer()

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class DrawingBoardHandler(tornado.websocket.WebSocketHandler):
    def on_message(self, message):
        message = json.loads(message)
        plt.imshow(message)
        pylab.show()
        image_array = np.zeros((1, 28, 28, 1), np.float32)
        for i in range(2,30):
            for j in range(2,30):
                temp = message[i][j][3]
                image_array[0][i-2][j-2][0] = temp/255
        plt.imshow(image_array[0,:,:,0], cmap='gray')
        pylab.show()
        res = int(recognizer.recognize(image_array))
        self.write_message(json.dumps(res))


url = [
    (r'/', IndexHandler),
    (r'/drawing-board', DrawingBoardHandler)
]

settings = {
    "template_path": os.path.join(os.path.dirname(__file__), "templates"),
    "debug": True,
}

application = tornado.web.Application(
    handlers = url,
    **settings
)


define("port", default=8080, help="run on the given port", type=int)

def main():
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    print("Development server is running at http://0.0.0.0:%s" % options.port)
    print("Quit the server with Control-C")
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    main()
