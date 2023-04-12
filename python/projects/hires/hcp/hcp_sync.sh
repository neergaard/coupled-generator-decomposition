aws="/mrhome/jesperdn/apps/aws/dist/aws"
src="s3://hcp-openaccess/HCP_1200"
dest="/mnt/projects/INN/jesper/nobackup/HiRes/hcp/data"
subjectlist=$dest/subjects.txt

while read subject; do # 'subject' is the variable name
    echo "Fetching data for $subject"
    #
    subdir=$subject/T1w/
    $aws s3 sync $src/$subdir $dest/$subdir --exclude="*" --include="T*w_acpc_dc_restore.nii.gz"

    # subdir=$subject/T1w/Native
    # $aws s3 sync $src/$subdir $dest/$subdir --exclude="*" --include="$subject.*.white.native.surf.gii" --include="$subject.*.pial.native.surf.gii"

    subdir=$subject/T1w/$subject/surf
    $aws s3 sync $src/$subdir $dest/$subdir --exclude="*" --include="*h.white" --include="*h.pial" --include="*h.curv" --include="*h.curv.pial" --include="*h.thickness"
    subdir=$subject/T1w/$subject/mri
    $aws s3 sync $src/$subdir $dest/$subdir --exclude="*" --include="norm.mgz"
done < $subjectlist