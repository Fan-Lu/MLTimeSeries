fid = fopen('fs.txt', 'w');
fprintf(fid, '%f\r\n', fs);
fclose(fid);