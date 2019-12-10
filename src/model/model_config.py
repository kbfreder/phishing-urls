#!/usr/bin/env python
clf_params = {'n_estimators' : 75,
                        'max_depth' : 10,
                        'criterion': 'entropy'}

model_cols = ['subdomain_null_ind', 'subdomain_www_ind', 'length_url',
              'domain_dot_cnt', 'path_dot_cnt', 'hostname_dash_cnt',
              'hostname_entropy', 'url_entropy', 'php_ind', 'abuse_ind',
              'admin_ind', 'verification_ind', 'length_path_frac_url_len',
              'length_domain_frac_url_len', 'url_slash_cnt_frac_url_len',
              'url_digit_cnt_frac_url_len', 'url_special_char_cnt_frac_url_len',
              'url_reserved_char_cnt_frac_url_len']
