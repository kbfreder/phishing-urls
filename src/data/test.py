'''test.py

Test script

Usage:
    test.py download <n-chunks> [options]
    test.py parse [options]
    test.py download <n-chunks> parse
    test.py [options]

Options
    -cols                   Columns to keep during parsing
    -v                      Verbose
    -h --help               Show docstring.
    -t                      Test mode.

'''


from docopt import docopt

arguments = docopt(__doc__, help=True)

def download(n_chunks):
    print('This is the download function, with n_chunks = ', n_chunks)

def parse_files(cols_to_keep=['ts', 'url', 'languages']):
    print('This would parse the files')
    print(cols_to_keep)

if __name__ == '__main__':
    # print(arguments)

    if arguments['-t']:
        print('Test mode!')

    if arguments['download']:
        download(arguments['<n-chunks>'])

    if arguments['parse']:
        if arguments['-cols'] is not None:
            parse_files()
        else:
            parse_files(arguments['-cols'])
