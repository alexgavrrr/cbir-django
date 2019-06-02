import xmlrpc.client


def run(
        port,
        database,
        name,
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

    result = s.search(database, name, query, search_params)
    print(f'Result: {result}')


if __name__ == '__main__':
    pass
