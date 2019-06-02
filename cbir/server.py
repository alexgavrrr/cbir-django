import sys
import socketserver
import time

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import xmlrpc.client


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


class RPCThreadingServer(socketserver.ThreadingMixIn, SimpleXMLRPCServer):
    pass


COUNTER = 0


def run(port, nthreads):
    with RPCThreadingServer(('localhost', port),
                            requestHandler=RequestHandler) as server:

        server.register_introspection_functions()

        server.register_function(pow)

        @server.register_function(name='add')
        def adder_function(x, y):
            return x + y

        @server.register_function
        def increment_counter(x):
            global COUNTER

            COUNTER += x
            return COUNTER

        @server.register_function
        def sleep(x):
            print(f'Sleep {x} sec...')
            time.sleep(x)
            return x

        @server.register_function
        def mul(x, y):
            return x * y

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")
            sys.exit(0)


if __name__ == '__main__':
    port = 8701
    if sys.argv[1] == 's':
        run(port, 4)
    elif sys.argv[1] == 'c':
        s = xmlrpc.client.ServerProxy(f'http://localhost:{8701}')
        print(s.increment_counter(1))
        print(s.system.listMethods())
    elif sys.argv[1] == 'cm':
        s = xmlrpc.client.ServerProxy(f'http://localhost:{8701}')
        count = int(sys.argv[2])
        for i in range(count):
            res = s.increment_counter(1)
            if i == 500:
                print(res)
        print(res)
    elif sys.argv[1] == 'cs':
        s = xmlrpc.client.ServerProxy(f'http://localhost:{8701}')
        count = int(sys.argv[2])
        s.sleep(count)
        print(f'Slept {count} sec')
