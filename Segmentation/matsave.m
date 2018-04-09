fid = fopen('test.txt', 'w');
fprintf(fid, '%f\r\n', test);
fclose(fid);

fid = fopen('test_fs.txt', 'w');
fprintf(fid, '%f\r\n', test_fs);
fclose(fid);

fid = fopen('noise.txt', 'w');
fprintf(fid, '%f\r\n', noise);
fclose(fid);

fid = fopen('noise_fs.txt', 'w');
fprintf(fid, '%f\r\n', noise_fs);
fclose(fid);