import csv
import os
def save(exp_no, desc, subject, modelname, min_val_loss, val_cc, epoch, results_csv, train_time, param_count, config, all_results_path):
    header = {'Sl_no':exp_no,
            'subject':subject,
            'model_name': modelname,
            'model_desc':desc,
            'val_loss': min_val_loss,
            'cc': val_cc,
            'epoch':epoch,
            'train_time':train_time,
            'param count':param_count,
            'encoder layers':config['in_fft_n_layers'],
            'encoder heads':config['in_fft_n_heads'],
            'encoder attn dim':config['in_fft_d_head'],
            'encoder ff dim':config['in_fft_conv1d_filter_size'],
            'encoder out dim':config['in_fft_output_size'],
            'decoder layers':config['out_fft_n_layers'],
            'decoder heads':config['out_fft_n_heads'],
            'decoder attn dim':config['out_fft_d_head'],
            'decoder ff dim':config['out_fft_conv1d_filter_size'],
            'decoder out dim':config['out_fft_output_size'],
            'all_result_numbers':all_results_path}

    check = os.path.exists(results_csv)
    file = open(results_csv, 'a', newline ='')
    with file:
        writer = csv.DictWriter(file, fieldnames=list(header.keys()))
        if not check: writer.writeheader()
        writer.writerow(header)
