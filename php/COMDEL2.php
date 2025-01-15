<?php
if(!isset($_POST["seqs"])){
    sleep(2);
    header("Content-Type: application/json;");
    echo '{"success":0,"mesage":"no post data"}';
    exit;
}
if(empty($_POST["seqs"])){
    sleep(2);
    header("Content-Type: application/json;");
    echo '{"success":0,"mesage":"post data is empty"}';
    exit;
}

// sequence pre-processing
$new_seq = preg_replace('/\s/', '', $_POST["seqs"]);
$seq_max_len = 2000;
$new_seq = strtoupper($new_seq);
if(preg_match('/[^AFCUDNEQGHLIKOMPRSTVWY]/', $new_seq, $matches)){
    sleep(2);
    header("Content-Type: application/json;");
    echo '{"success":0,"mesage":"Illegal amino acid character"}';
    exit;
}
if(strlen($new_seq)>$seq_max_len){
    sleep(2);
    header("Content-Type: application/json;");
    echo '{"success":0,"mesage":"amino acid max length=2000aa"}';
    exit;
}

# save function
function save_temp_file($name, $content){
    $file_ = fopen($name, "w");
    fwrite($file_, $content);
    fclose($file_);
}

#echo "hello";
#exit;

#---------------------------
#set key var
$temp_file_dir = "/var/www/html/log/";#"/var/log/CT_AMP/"; # save file from uploading by user
$seqs = $new_seq; # get seqs from user
$seqs = sprintf("%s\nseq1,%s\n","Name,Sequence",$seqs);
$user_md5_str = $_SERVER['REMOTE_ADDR']."-".date('Y-m-d H:i:s').$seqs; # ip+date to be temp name
$user_md5_hash = md5($user_md5_str);
$user_work_dir = $temp_file_dir.$user_md5_hash."/";
$user_results_dir = $user_work_dir."results/";
$user_plt_results_dir = $user_work_dir."plt_results/";;

// create dir for user analyse
// $user_md5_str = "9216aa520bc648bbb94ae69a1bf37c3e";
mkdir($user_work_dir, 0777, false);
mkdir($user_results_dir, 0777, false);
mkdir($user_plt_results_dir, 0777, false);
$user_file_path = $user_work_dir ."user.csv"; # md5 it
save_temp_file($user_file_path, $seqs); # save file uploaded by user

// --------------------------------------
// start analyse
// step 1
// chdir("COMDEL");
// python COMDEL2_AMP.py -infile %s -plt_results %s -
$COMDEL2_work_dir = "/home/lw/COMDEL2/predict";
$COMDEL1_work_dir = "/home/lw/COMDEL2/COMDEL";
chdir($COMDEL1_work_dir);
$command = sprintf("python3 COMDEL2_AMP.py -cuda -infile %s -results %s -plt_results %s", $user_file_path, $user_results_dir, $user_plt_results_dir);
exec($command,$out);

// step 2
// chdir("predict");
// python predict.py --infile test.csv --result_path results --plt_path plt_results
chdir($COMDEL2_work_dir);
$command = sprintf("python3 predict.py --infile %s --result_path %s --plt_path %s", $user_file_path, $user_results_dir, $user_plt_results_dir);
exec($command,$out);
#print_r($out);
#exit;

// step3 heat map
chdir($COMDEL2_work_dir);
$heat_map_in_file = sprintf("%s/Combined_predict.csv",$user_results_dir);
$heat_map_out_file = sprintf("%s/predseq_circular_heatmap.png",$user_plt_results_dir);
$command = sprintf("Rscript pred_seq_circular.R %s %s", $heat_map_in_file, $heat_map_out_file);
exec($command, $out);

// step zip file
$zip_dir = $user_work_dir;
chdir($zip_dir);
exec("zip -r results.zip ./");
$zip_file_path = $zip_dir."/results.zip";
$zip_file_path = str_replace('\\', '/', $zip_file_path);
$zip_url = sprintf("/log/%s/results.zip",$user_md5_hash);

header("Content-Type: application/json;");
$url_img_prefix =  "/log/".$user_md5_hash;
$imgs = array(
    0=>sprintf("%s/plt_results/ACP_predictions.png",$url_img_prefix),
    1=>sprintf("%s/plt_results/ADP_predictions.png",$url_img_prefix),
    2=>sprintf("%s/plt_results/AGP_predictions.png",$url_img_prefix),
    3=>sprintf("%s/plt_results/AHP_predictions.png",$url_img_prefix),
    4=>sprintf("%s/plt_results/AIP_predictions.png",$url_img_prefix),
    5=>sprintf("%s/plt_results/AMP_predictions.png",$url_img_prefix),
    6=>sprintf("%s/plt_results/BiP_predictions.png",$url_img_prefix),
    7=>sprintf("%s/plt_results/CPP_predictions.png",$url_img_prefix),
    8=>sprintf("%s/plt_results/DDP_predictions.png",$url_img_prefix),
    9=>sprintf("%s/plt_results/DeP_predictions.png",$url_img_prefix),
    10=>sprintf("%s/plt_results/HeP_predictions.png",$url_img_prefix),
    11=>sprintf("%s/plt_results/NuP_predictions.png",$url_img_prefix),
    12=>sprintf("%s/plt_results/UmP_predictions.png",$url_img_prefix),
    13=>sprintf("%s/plt_results/predseq_circular_heatmap.png",$url_img_prefix)
);
$results = array(
    "success"=>1,
    "imgs"=>$imgs, 
    "file_csv"=>"csv",
    "seqs"=>$new_seq,
    "zip_url"=>$zip_url
);
sleep(2);
echo json_encode($results);
?>
