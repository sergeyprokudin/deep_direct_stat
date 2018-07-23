#!/usr/bin/env bash

dbash::pp() {
    echo -e "$1"
}

dbash::user_confirm() {
    NOT_FINISHED=true
    while ${NOT_FINISHED} ;do
        echo -e -n "$1 [y/n] default($2) "
        read USER_INPUT;
        if [[ "y" == "${USER_INPUT}" ]];then
            USER_CONFIRM_RESULT="y";
            NOT_FINISHED=false;
        elif [[ "n" == "${USER_INPUT}" ]];then
            USER_CONFIRM_RESULT="n";
            NOT_FINISHED=false;
        elif [[ "" == "${USER_INPUT}" ]];then
            USER_CONFIRM_RESULT="$2";
            NOT_FINISHED=false;
        else
            echo -e "# only y, n, and nothing, are possible choices."
            echo -e "# default is $2"
        fi
    done
}
