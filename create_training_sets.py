import os
import random
import chardet
import hashlib
import re

def strip_comments(data):
    res = re.sub(r'<#.*?#>', '\n', data, flags=re.MULTILINE|re.DOTALL)
    res = re.sub(r'#.*$', '', res, flags=re.MULTILINE)
    res = re.sub(r'[\r\n]\s*[\r\n]', '\n', res, flags=re.MULTILINE) 
    return res

def md5(data):
    m = hashlib.md5()  
    m.update(data)
    return m.hexdigest()

def decode(path):
    with open(path, 'rb') as inf: 
        data = inf.read() 
        encoding = chardet.detect(data)['encoding'] 
        return data.decode(encoding)

def load_paths(dirnames):
    paths = [] 
    for dirname in dirnames:
        for (dirpath, _, fnames) in os.walk(dirname): 
            for fname in fnames: 
                path = os.path.join(dirpath, fname) 
                paths.append(path)
    return paths

def create_combined_file(paths, maxlen, outfilename):
    seen = set()
    random.shuffle(paths)
    skipped = 0
    errors = 0
    
    rawbytes = ''
    for path in paths:
        try:
            data = decode(path)
            data = strip_comments(data)
            md5sum = md5(data.encode('utf-8'))
            if md5sum in seen:
                skipped += 1
                continue
            seen.add(md5sum)

            rawbytes += data
            rawbytes += '<|endoftext|>'
        except Exception as e:
            print(f'Error on {path} ({e})')
            errors += 1
        if len(rawbytes) > maxlen:
            break
    
    with open(outfilename, 'w') as outf:
        print(f'Writing {len(rawbytes)} bytes to {outfilename} ...')
        outf.write(rawbytes)

    print(f'Included {len(seen)} files, Skipped {skipped} files b/c they were duplicates, errored out on {errors} files.')

def create_binary_training_set(paths, maxfiles, outpath):
    seen = set()
    random.shuffle(paths)
    skipped = 0
    errors = 0
    
    for path in paths:
        try:
            data = decode(path)
            data = strip_comments(data)
            md5sum = md5(data.encode('utf-8'))
            if md5sum in seen:
                skipped += 1
                continue
            seen.add(md5sum)
            
            with open(f'{outpath}/{md5sum}.txt', 'w') as outf:
                outf.write(data)
            
            if len(seen) >= maxfiles:
                break
        except Exception as e:
            print(f'Error on {path} ({e})')
            errors += 1
    print(f'Included {len(seen)} files, Skipped {skipped} files b/c they were duplicates, errored out on {errors} files.')

if __name__ == '__main__':
    # TODO: command line args for basepath
    # TODO: consider sampling without replacement for finetuning validation sets (may not matter)

    basepath = '/Users/jasontrost/data/PowerShell'
    finetuning = f'{basepath}/finetuning'
    classification = f'{basepath}/classification'

    paths = load_paths([f'{basepath}/Github', f'{basepath}/Technet', f'{basepath}/PowerShellGallery'])
    create_combined_file(paths, 50*1024*1024, f'{finetuning}/training/Normal-sample-training.txt')
    create_combined_file(paths, 1024*1024, f'{finetuning}/validation/Normal-sample-validation.txt')
    create_binary_training_set(paths, 10000, f'{classification}/benign')

    paths = load_paths([f'{basepath}/InvokeObfuscation'])
    create_combined_file(paths, 50*1024*1024, f'{finetuning}/training/InvokeObfuscation-sample-training.txt')
    create_combined_file(paths, 1024*1024, f'{finetuning}/validation/InvokeObfuscation-sample-validation.txt')
    create_binary_training_set(paths, 10000, f'{classification}/obfuscated')

    paths = load_paths([f'{basepath}/PowerShell-from-VT'])
    create_combined_file(paths, 50*1024*1024, f'{finetuning}/training/PowerShell-from-VT-sample-training.txt')
    create_combined_file(paths, 1024*1024, f'{finetuning}/validation/PowerShell-from-VT-sample-validation.txt')
    create_binary_training_set(paths, 10000, f'{classification}/malicious')

