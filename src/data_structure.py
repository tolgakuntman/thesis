# Table0 Repository info
repo_columns = [
    'repo_url',
    'repo_name',
    'owner',
    'description',
    'date_created',
    'date_last_push',
    'homepage',
    'repo_language',
    'forks_count',
    'stars_count',
    'size',
    'topics',
    'open_issues_count',
    'total_commits',
    'contributors_count',
    'releases_count',
    'tags_count',
    'collecting_date'
]

# Table1 CVE_CWE_FC_VCC
cve_cwe_commit_columns = [
    'cve_id',
    'cwe_id',
    'repo_url',
    'fc_hash',
    'vcc_hash'
]

# Table2 FC(Fix Commit) info
# Table3 VCC(Vulnerability Contribute Commit) info
commit_columns = [
    'hash',
    'commit_type',  # 'VCC' or 'FC'
    'msg',
    'author',
    'author_date',
    'author_timezone',
    'committer',
    'committer_date',
    'committer_timezone',
    'in_main_branch',
    'merge',
    'parents',
    'num_lines_deleted',
    'num_lines_added',
    'num_lines_changed',
    'num_files_changed',
    'dmm_unit_size',
    'dmm_unit_complexity',
    'dmm_unit_interfacing'
]

# Table4 (changed)file info (remove all annotations), may have duplication
file_columns = [
    'hash',    # -- FK to file hash
    'commit_label',  # NEW: 'VCC' or 'FC' to distinguish commit type
    'filename',
    'old_path',
    'new_path',
    'change_type',
    'diff',
    'diff_parsed',
    'num_lines_added',
    'num_lines_deleted',
    'code_after',
    'code_before',
    'num_method_changed',
    'num_lines_of_code',
    'complexity',
    'token_count',
    # 'programming_language',
]

# Table5 (changed)function info (remove all annotations), may have duplication
function_columns = [
    'hash',
    'commit_label',  # NEW: 'VCC' or 'FC' to distinguish commit type
    'name',
    'filename',
    'num_lines_of_code',
    'complexity',
    'token_count',
    'parameters',
    'signature',
    'start_line',
    'end_line',
    # 'fan_in',
    # 'fan_out',
    # 'general_fan_out', Still under developing, not working in lizard
    'length',
    'top_nesting_level',
    'code',
    'before_change'
]

# Table6 issue_info
issue_columns = [
    'issue_id', # int: PK1 issue id from github        
    'fc_hash',  # str: FK to commit_info.fc_hash
    'vcc_hash',  # str: FK to commit_info.vcc_hash
    'opened_by_dev_id', # int: github id of the dev: FK to developer_info.dev_id
    'created_at', # datetime: issue created at
    'closed_at', # datetime: issue closed at
    'labels',   # list: labels associated with the issue
    'window_since', # datetime: start of the filtering window
    'window_until', # datetime: end of the filtering window
    'matched_anchor', # enum:FC or VCC or both
]

# Table7 pull_request_info
pull_request_columns = [
    'pr_id', # int: PK1 pull request id from github
    'fc_hash',  # str: FK to commit_info.fc_hash
    'vcc_hash',  # str: FK to commit_info.vcc_hash
    'pr_url', # str: url of the pull request
    'opened_by_dev_id', # int: github id of the dev: FK to developer_info.dev_id
    'closed_by_dev_id', # int: github id of the closer dev: FK to developer_info.dev_id
    'created_at', # datetime: pull request created at
    'closed_at', # datetime: pull request closed at
    'state', # enum: open, closed
    'title', # str: title of the pull request 
    'window_since', # datetime: start of the filtering window
    'window_until', # datetime: end of the filtering window
    'matched_anchor', # enum:FC or VCC or both
]

# Table8 release_tag
release_tag_columns = [
    'tag_id', # PK: unique tag identifier from github
    'tag_name', # str: name of the tag   
    'fc_hash',  # str: FK to commit_info.fc_hash
    'vcc_hash',  # str: FK to commit_info.vcc_hash
    'created_at', # datetime: tag created at
    'window_since', # datetime: start of the filtering window
    'window_until', # datetime: end of the filtering window
    'matched_anchor', # enum:FC or VCC or both 
]

# Table9 github_label (label definitions in a repo)
github_label_columns = [
    'label_id', # PK: unique label identifier from github 
    'label_name', # str: name of the label : FK to issue_info.labels           
    'color', # hex: color code of the label           
    'description', # str: description of the label    
    'is_default', # bool: whether the label is default or custom     
    'collecting_date', # datetime: when the label data was collected
]

# Table10 developer_info (developer statistics)
# Note: dev_id can be email (from commits) or GitHub user ID (from issues/PRs)
developer_info_columns = [
    'dev_id',              # PK: email or GitHub user ID (see dev_id_type)
    'dev_id_type',         # enum: 'email' or 'github_user_id' -- we can link the email-based devs to commits
    'github_user_id',      # GitHub user ID (if available)
    'first_seen_at',       # datetime: first commit/activity
    'total_commits',       # int: total commits as author
    'active_weeks',        # int: number of weeks with activity
    'total_issues',        # int: issues opened (GitHub ID only)
    'total_pull_requests', # int: PRs opened (GitHub ID only)
]

# Table11 commit_author (junction table linking commits to developers)
commit_author_columns = [
    'commit_hash',         # FK -> commit_info.hash
    'role',                # 'author' | 'committer'
    'dev_id',              # FK -> developer_info.dev_id (email)
]

# Table12 ownership_window (file ownership over time windows before VCC date)
# 
# SEMANTIC DEFINITIONS:
# - ownership_ratio, lines_owned: git blame AT anchor (who wrote the code)
# - edits_in_window, lines_added/deleted: activity BEFORE anchor (who was modifying)
# - dev_id: SHA256 hash of normalized email for privacy
#
# Windows: 30, 90, 180 days before vulnerability was introduced
ownership_window_columns = [
    'commit_hash',              # Anchor commit (VCC or FC hash) -- FK to commit_info.hash  --PK1 
    'file_path',                # Resolved file path at anchor  -- FK to file_info.new_path --PK1
    'dev_id',                   # PK3: SHA256 hash of normalized email -- can be linked to dev info table
    'ownership_ratio',          # float: blame ownership at anchor (0-1)
    'lines_owned',              # int: lines owned at anchor
    'edits_in_window',          # int: commits by developer before anchor
    'lines_added_in_window',    # int: lines added before anchor
    'lines_deleted_in_window',  # int: lines deleted before anchor
    'total_lines',              # int: total lines in file at anchor
    'window_days',              # enum: PK2: 30, 90, or 180 days
    'window_start',             # date: anchor - window_days (ISO format)
    'window_end',               # date: anchor commit date (ISO format)
]


"""Features Explanation: fan_in: This metric measures how many other functions or methods call a particular function. 
In other words, it indicates the number of functions that depend on the function in question. fan_out: This metric 
measures how many functions or methods are called by a particular function. It indicates the number of functions that 
the function in question depends on. general_fan_out: While fan_out covers direct dependencies (i.e., the functions a 
method directly calls), general_fan_out might refer to a broader measure of dependencies. This can include not just 
direct calls but also transitive dependencies — essentially, the total number of unique functions that can be reached 
starting from the function in question, including all levels of indirect calls."""