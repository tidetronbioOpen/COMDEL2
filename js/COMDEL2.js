
function example_click(){
    $("#elementor-search-form-20ddbd4").val("AGSTLIFYCHDE");
}

function waitBlock(){
    $(".wait-block").css({'transform': 'rotate(' + (++degree % 360) + 'deg)'});
    degree = degree>=360?0:degree
}
var degree = 0;
var order_id = null;
var is_uploaded = false;

$(function(){
    setInterval(waitBlock, 10);
    $("#btn-example").click(example_click);
})


function showResult(data){
    console.log(data);
    console.log(data["imgs"]);
    var frame_tag = $(".anti-peptides-prob");
    frame_tag.empty();
    for(var i=0;i<data["imgs"].length-1;i++){
        var tag = $("#anti-peptides-prob-templete").clone();
        tag.attr({
            "id":"anti-peptides-prob-img",
            "src":data["imgs"][i]
        });
        frame_tag.append(tag);
        tag.fadeIn(1000);
    }
    // heat
    var tag_heat = $("#anti-peptides-heat-item");
    tag_heat.attr({
        "src":data["imgs"][data["imgs"].length-1]
    });
    tag_heat.fadeIn(1000);
    $("#anti-peptides-frame").fadeIn(1000);
    
}

function scrollAction(end_times){
    end_times+=1;
    var cur_height = window.scrollY;
    console.log(window.scrollY);
    window.scrollTo(0, cur_height);
    window.scrollTo(0, cur_height+end_times*50);
    if(end_times>=4) return;
    if(cur_height+end_times*50>=400) return;
    setTimeout(()=>{scrollAction(end_times)},100);
}

// timer
function ask_task(data){
    // 
    if(data["status"]=="done"){
        is_uploaded = false;
        $(".elementor-counter-title").text("");
        $(".wait-block-mes").text("");
        $(".wait-block").css("display","none");
        $(".zip-results-link").click(function(){
            window.open(data["zip_url"], '_blank');
        });
        $("#upload-seqs").text(data["seqs"]);
        showResult(data);
        return
    }
    if(data["status"]=="error"){
        is_uploaded = false;
        $(".wait-block").css("display","none");
        return 
    }
    $.ajax({
        url:"/comdel2_pred",
        type:"POST",
        data:{"req_type":"ask","order_id":order_id},
        success:function(res){
            // $(".elementor-counter-title").text(data["mes"]);
            $(".wait-block-mes").text(data["mes"]);
            setTimeout(function(){ask_task(res)},5000);
        },
        error:function(res){
            console.error(res);
        },
    });
}


function subfun(){
    if(is_uploaded){
        $(".elementor-counter-title").text("You have submitted");
        return false;
    }
    is_uploaded  = true;
    $("#anti-peptides-frame").fadeOut(1000);
    $(".wait-block").show();
    $(".elementor-counter-title").text("");
    var my_data = $('.elementor-search-form').serialize();
    var data = `req_type=submit&${my_data}`;
    console.log(data);
    scrollAction(0);
    $.ajax({
        url : "/comdel2_pred",
        type : "POST",
        data : data,
        success : function(data) {
            // $(".wait-block").hide();
            console.log("->",data);
            // success = false
            if(data["success"]==0){
                is_uploaded = false;
                $(".elementor-counter-title").text(data["mes"]);
                $(".wait-block").css("display","none");
                return
            }
            // success add in task line
            if(data["status"]=="add_success"){
                order_id = data["order_id"];
            }
            if(data["status"]!="done"){
                ask_task(data);
            }
            
            // console.log(data);
        },
        error : function(data) {
            // console.warn(data);
            $(".wait-block").css("display","none");
            console.warn("warning");
        }
    });
    return false;
}
