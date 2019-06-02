import logging
import sys
import socketserver
import time
from collections import defaultdict

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import xmlrpc.client

from cbir.cbir_core import CBIRCore


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


class RPCThreadingServer(socketserver.ThreadingMixIn, SimpleXMLRPCServer):
    pass


CBIR_CORES = defaultdict(lambda: defaultdict(lambda: None))


def run(port, nthreads):
    with RPCThreadingServer(('localhost', port),
                            requestHandler=RequestHandler) as server:

        server.register_introspection_functions()

        @server.register_function
        def search(database, name, query, search_params):
            logger = logging.getLogger()
            cbir_core = CBIR_CORES[database][name]
            if cbir_core is None:
                cbir_core = CBIRCore.get_instance(database, name)

                start = time.time()
                logger.info(f'Loading fd, ca, bow, inv for {database}-{name}...')
                cbir_core.set_fd(cbir_core.load_fd())
                cbir_core.set_ca(cbir_core.load_ca())
                cbir_core.set_bow(cbir_core.load_bow())
                cbir_core.set_inv(cbir_core.load_inv())
                CBIR_CORES[database][name] = cbir_core
                time_loading = round(time.time() - start, 3)
                logger.info(f'Loaded fd, ca, bow, inv for {time_loading} sec.')
            else:
                logger.info(f'cbir_core {database}-{name} already loaded')

            result = cbir_core.search(query, **search_params)
            return result

        print(f'Listening on port {port}')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, exiting.")
            sys.exit(0)


if __name__ == '__main__':
    if sys.argv[1] == 's':
        pass
    elif sys.argv[1] == 'c':
        pass
