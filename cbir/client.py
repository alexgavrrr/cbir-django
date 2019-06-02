import xmlrpc.client


def run(
        port,
        database,
        index,
        query,
        topk,
        sv,
        qe,
        **kwargs, ):

    s = xmlrpc.client.ServerProxy(f'http://localhost:{port}', allow_none=True)
    search_params = {
        'topk': topk,
        'sv_enable': sv,
        'qe_enable': qe,
        'n_candidates': 100,
    }

    result = s.search(database, index, query, search_params)
    print(f'Result: {result}')


if __name__ == '__main__':
    pass
